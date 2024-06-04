"""
SPQE classes
====================================
Classes for implementing the selected variant of the projetive quantum eigensolver
"""

import qforte as qf

from qforte.abc.uccpqeabc import UCCPQE
from qforte.experiment import *
from qforte.utils.transforms import *
from qforte.utils.state_prep import *
from qforte.utils.trotterization import trotterize
from qforte import find_irrep
from qforte.utils import moment_energy_corrections
from qforte.maths import optimizer

import numpy as np


class SPQE(UCCPQE):
    """This class implements the selected projective quantum eigensolver (SPQE) for
    disentagled UCC like ansatz.
    In SPQE, a batch of imporant particle-hole operators
    :math:`\\{ e^{t_\\mu (\\hat{\\tau}_\\mu - \\hat{\\tau}_\\mu^\\dagger )} \\}` are
    added at each macro-iteration :math:`n` to the SPQE unitary :math:`\\hat{U}(\\mathbf{t})`,
    wile all current parameters are optimized using the quasi-Newton PQE update
    with micro-iterations :math:`k`.

    In our selection approach we consider a (normalized) quantum state of the form

    .. math::
        | \\tilde{r} \\rangle  = \\tilde{r}_0 | \\Phi_0 \\rangle + \\sum_\\mu \\tilde{r}_\\mu  | \\Phi_\\mu \\rangle

    where the quantities :math:`\\tilde{r}_\\mu` are approximately proportional to
    the residuals :math:`r_\\mu`.
    The state :math:`| \\tilde{r} \\rangle` can be approximately reproduced via

    .. math::
        | \\tilde{r} \\rangle \\approx \\hat{U}^\\dagger e^{i \\Delta t \\hat{H}} \\hat{U} | \\Phi_0 \\rangle

    .. math::
        \\approx (1 + i\\Delta t \\hat{U}^\\dagger \\hat{H} \\hat{U})  | \\Phi_0 \\rangle + \\mathcal{O}(\\Delta t^2).

    We note that in this implementation we use a Trotter approximation for the time
    evolution unitary.
    Measuring :math:`\\langle \\hat{Z} \\rangle` for each qubit yields a bitstring
    that has corresponding determinat and operator
    :math:`(\\hat{\\tau}_\\mu - \\hat{\\tau}_\\mu^\\dagger )`
    with probablility proportional to :math:`|\\tilde{r}_\\mu|^2`.
    The operators corresponding to the largest :math:`|\\tilde{r}_\\mu|^2` values
    are then added to :math:`\\hat{U}(\\mathbf{t})` at each macro-iteration.
    """

    def run(
        self,
        spqe_thresh=1.0e-2,
        spqe_maxiter=20,
        dt=0.001,
        M_omega="inf",
        opt_thresh=1.0e-5,
        opt_maxiter=30,
        use_cumulative_thresh=True,
        max_excit_rank=None,
        optimizer="Jacobi",
    ):
        """
        spqe_thresh: float
            The convergence threshold against which the norm of the residual is compared
            (in L2 norm) to determine macroiteration convergence.
        spqe_maxiter: int
            The maximum number of macroiterations
        dt: float
            The coefficient of an "imaginary time" step used for estimating the residual,
            eq. 15/16 of 10.1103/PRXQuantum.2.030301. Smaller values are more accurate
            in the infinite measurement limit, but approach their limit more slowly.
        M_omega: Union['inf', float]
            The number of measurements to use when constructing the estimated residual, eq. 15/16
            of 10.1103/PRXQuantum.2.030301.
        opt_thresh: float
            The convergence threshold used during microiterations.
        opt_maxiter: int
            The maximum number of microiterations
        use_cumulative_thresh: bool
            In brief, use True to add multiple operators per macroiteration, and False for one
            operator per macroiteration. When True, operators are added using the cumulative
            threshold as described in 10.1103/PRXQuantum.2.030301 Section 2E.
        max_excit_rank: Union[int, None]
            The maximum rank of excitations considered. Select None for no restrictions.
        optimizer: {'jacobi', 'nelder-mead', 'powell', 'bfgs', 'l-bfgs-b', 'cg', 'slsqp'}
            The optimizer to solve the residual equations.
        """

        if self._state_prep_type != "occupation_list":
            raise ValueError(
                "SPQE implementation can only handle occupation_list Hartree-Fock reference."
            )

        self._spqe_thresh = spqe_thresh
        self._spqe_maxiter = spqe_maxiter
        self._dt = dt
        if M_omega != "inf":
            self._M_omega = int(M_omega)
        else:
            self._M_omega = M_omega

        self._use_cumulative_thresh = use_cumulative_thresh
        self._optimizer = optimizer
        self._opt_thresh = opt_thresh
        self._opt_maxiter = opt_maxiter

        # _nbody_counts: list that contains the numbers of singles, doubles, etc. incorporated in the final ansatz
        self._nbody_counts = []
        self._n_classical_params_lst = []

        self._results = []
        self._energies = []
        self._grad_norms = []
        self._tops = []
        self._tamps = []
        self._stop_macro = False
        self._converged = False
        self._res_vec_evals = 0
        self._res_m_evals = 0

        self._curr_energy = 0.0

        # Resource estimates.
        self._n_classical_params = 0
        self._n_cnot = 0
        self._n_cnot_lst = []
        self._n_pauli_trm_measures = 0
        self._n_pauli_trm_measures_lst = []
        self._Nm = []

        # self._eiH: QuantumCircuit
        #     Used to estimate the residuals outside the zero'd set when selecting new residuals to zero.
        self._eiH, _ = trotterize(
            self._qb_ham,
            factor=self._dt * (0.0 + 1.0j),
            trotter_number=self._trotter_number,
        )

        for occupation in self._ref:
            if occupation:
                self._nbody_counts.append(0)

        # create a pool of particle number, Sz, and spatial symmetry adapted second quantized operators
        # Encode the occupation list into a bitstring
        ref = sum([b << i for i, b in enumerate(self._ref)])
        # `& mask_alpha` gives the alpha component of a bitstring. `& mask_beta` does likewise.
        mask_alpha = 0x5555555555555555
        mask_beta = mask_alpha << 1
        nalpha = sum(self._ref[0::2])
        nbeta = sum(self._ref[1::2])

        if max_excit_rank is None:
            max_excit_rank = nalpha + nbeta
        elif not isinstance(max_excit_rank, int) or max_excit_rank <= 0:
            raise TypeError(
                "The maximum excitation rank max_excit_rank must be a positive integer!"
            )
        elif max_excit_rank > nalpha + nbeta:
            max_excit_rank = nalpha + nbeta
            print(
                "\nWARNING: The entered maximum excitation rank exceeds the number of particles.\n"
                "         Proceeding with max_excit_rank = {0}.\n".format(
                    max_excit_rank
                )
            )
        self._pool_type = max_excit_rank

        idx = 0
        # Given a coefficient index, what is the index of the "corresponding" pool element? Used to compute the operator to add to the ansatz in macroiterations.
        self._coeff_idx_to_pool_idx = {}
        self._indices_of_zeroable_residuals_for_pool = set()
        self._pool_obj = qf.SQOpPool()
        for I in range(1 << self._nqb):
            alphas = [int(j) for j in bin(I & mask_alpha)[2:]]
            betas = [int(j) for j in bin(I & mask_beta)[2:]]
            # Enforce particle number and Sz symmetry
            if sum(alphas) == nalpha and sum(betas) == nbeta:
                # Enforce point group symmetry
                if (
                    find_irrep(
                        self._sys.orb_irreps_to_int,
                        [len(alphas) - i - 1 for i, x in enumerate(alphas) if x]
                        + [len(betas) - i - 1 for i, x in enumerate(betas) if x],
                    )
                    == self._irrep
                ):
                    # Create the bitstring of created/annihilated orbitals
                    excit = bin(ref ^ I).replace("0b", "")
                    # Confirm excitation number is non-zero
                    if excit != "0":
                        # Consider operators with rank <= max_excit_rank
                        if int(excit.count("1") / 2) <= self._pool_type:
                            occ_idx = [
                                int(i)
                                for i, j in enumerate(reversed(excit))
                                if int(j) == 1 and self._ref[i] == 1
                            ]
                            unocc_idx = [
                                int(i)
                                for i, j in enumerate(reversed(excit))
                                if int(j) == 1 and self._ref[i] == 0
                            ]
                            sq_op = qf.SQOperator()
                            sq_op.add(+1.0, unocc_idx, occ_idx)
                            sq_op.add(-1.0, occ_idx[::-1], unocc_idx[::-1])
                            sq_op.simplify()
                            self._pool_obj.add_term(0.0, sq_op)
                            self._coeff_idx_to_pool_idx[I] = idx
                            self._indices_of_zeroable_residuals_for_pool.add(I)
                            idx += 1

        # Given a pool index, what is the coefficient of the "corresponding" coefficient vector element? Used to extract significant residuals in microiterations.
        # WARNING! To support repeated operators, either replace this variable or have repeated operators in the pool (which seems an awful hack).
        self._pool_idx_to_coeff_idx = {
            value: key for key, value in self._coeff_idx_to_pool_idx.items()
        }

        self.print_options_banner()

        self.build_orb_energies()

        if self._max_moment_rank:
            print("\nConstructing Moller-Plesset and Epstein-Nesbet denominators")
            self.construct_moment_space()

        self._spqe_iter = 1

        if self._print_summary_file:
            f = open("summary.dat", "w+", buffering=1)
            f.write(
                f"#{'Iter(k)':>8}{'E(k)':>14}{'N(params)':>17}{'N(CNOT)':>18}{'N(measure)':>20}\n"
            )
            f.write(
                "#-------------------------------------------------------------------------------\n"
            )

        while not self._stop_macro:
            self.update_ansatz()

            if self._converged:
                break

            if self._verbose:
                print("\ntoperators included from pool: \n", self._tops)
                print("\ntamplitudes for tops: \n", self._tamps)

            self.solve()

            if self._max_moment_rank:
                print("\nComputing non-iterative energy corrections")
                self.compute_moment_energies()

            if self._verbose:
                print("\ntamplitudes for tops post solve: \n", np.real(self._tamps))
            if self._print_summary_file:
                f.write(
                    f"  {self._spqe_iter:7}    {self._energies[-1]:+15.9f}    {len(self._tamps):8}        {self._n_cnot_lst[-1]:10}        {sum(self._n_pauli_trm_measures_lst):12}\n"
                )
            self._spqe_iter += 1

        if self._print_summary_file:
            f.close()

        self._Egs = self._energies[-1]

        print("\n\n")
        print("---> Final n-body excitation counts in SPQE ansatz <---")
        print("\n")
        print(f"{'Excitaion order':>20}{'Number of operators':>30}")
        print("---------------------------------------------------------")
        for l, nl in enumerate(self._nbody_counts):
            print(f"{l+1:12}              {nl:14}")

        print("\n\n")
        if not self._max_moment_rank:
            print(
                f"{'Iter(k)':>8}{'E':>14}{'N(params)':>17}{'N(CNOT)':>18}{'N(measure)':>20}"
            )
            print(
                "-------------------------------------------------------------------------------"
            )

            for k, Ek in enumerate(self._energies):
                print(
                    f" {k:7}    {Ek:+15.9f}    {self._n_classical_params_lst[k]:8}        {self._n_cnot_lst[k]:10}        {sum(self._n_pauli_trm_measures_lst[:k+1]):12}"
                )

        else:
            print(
                f"{'Iter':>8}{'E':>14}{'E_MMCC(MP)':>24}{'E_MMCC(EN)':>19}{'N(params)':>14}{'N(CNOT)':>18}{'N(measure)':>20}"
            )
            print(
                "-----------------------------------------------------------------------------------------------------------------------"
            )

            for k, Ek in enumerate(self._energies):
                print(
                    f" {k+1:7}    {Ek:+15.9f}    {self._E_mmcc_mp[k]:15.9f}    {self._E_mmcc_en[k]:15.9f}    {self._n_classical_params_lst[k]:8}        {self._n_cnot_lst[k]:10}        {sum(self._n_pauli_trm_measures_lst[:k+1]):12}"
                )

        self._n_classical_params = len(self._tamps)
        self._n_cnot = self._n_cnot_lst[-1]
        self._n_pauli_trm_measures = sum(self._n_pauli_trm_measures_lst)

        self.print_summary_banner()
        self.verify_run()

    def run_realistic(self):
        raise NotImplementedError("run_realistic() is not fully implemented for SPQE.")

    def verify_run(self):
        self.verify_required_attributes()
        self.verify_required_PQE_attributes()
        self.verify_required_UCCPQE_attributes()

    def print_options_banner(self):
        print("\n-----------------------------------------------------")
        print("       Selected Projective Quantum Eigensolver   ")
        print("-----------------------------------------------------")

        print("\n\n               ==> SPQE options <==")
        print("---------------------------------------------------------")

        self.print_generic_options()

        print("Use qubit excitations:                   ", self._qubit_excitations)
        print("Use compact excitation circuits:         ", self._compact_excitations)

        opt_thrsh_str = "{:.2e}".format(self._opt_thresh)
        spqe_thrsh_str = "{:.2e}".format(self._spqe_thresh)
        print("Optimizer:                               ", self._optimizer)
        if self._diis_max_dim >= 2 and self._optimizer.lower() == "jacobi":
            print("DIIS dimension:                          ", self._diis_max_dim)
        else:
            print("DIIS dimension:                           Disabled")
        print("Maximum number of micro-iterations:      ", self._opt_maxiter)
        print("Micro-iteration residual-norm threshold: ", opt_thrsh_str)
        print("Maximum excitation rank in operator pool:", self._pool_type)
        print("Number of operators in pool:             ", len(self._pool_obj))
        print("Macro-iteration residual-norm threshold: ", spqe_thrsh_str)
        print("Maximum number of macro-iterations:      ", self._spqe_maxiter)

    def print_summary_banner(self):
        print("\n\n                ==> SPQE summary <==")
        print("-----------------------------------------------------------")
        print("Final SPQE Energy:                           ", round(self._Egs, 10))
        if self._max_moment_rank:
            print(
                "Moment-corrected (MP) SPQE Energy:           ",
                round(self._E_mmcc_mp[-1], 10),
            )
            print(
                "Moment-corrected (EN) SPQE Energy:           ",
                round(self._E_mmcc_en[-1], 10),
            )
        print("Number of operators in pool:                 ", len(self._pool_obj))
        print("Final number of amplitudes in ansatz:        ", len(self._tamps))
        print("Number of classical parameters used:         ", self._n_classical_params)
        print("Number of CNOT gates in deepest circuit:     ", self._n_cnot)
        print(
            "Number of Pauli term measurements:           ", self._n_pauli_trm_measures
        )
        print("Number of residual vector evaluations:       ", self._res_vec_evals)
        print("Number of individual residual evaluations:   ", self._res_m_evals)

    def get_residual_vector(self, trial_amps):
        """
        Input
        -----
        trial_amps list[float]
         The trial value of the parameters, in order corresponding to the ansatz parameters.

        Output

        list[float]
         The residuals, in order corresponding to the ansatz parameters.
        """

        assert len(trial_amps) == len(self._tops)
        U = self.ansatz_circuit(trial_amps)
        qc_res = qforte.Computer(self._nqb)
        qc_res.apply_circuit(self._Uprep)
        qc_res.apply_circuit(U)
        qc_res.apply_operator(self._qb_ham)
        qc_res.apply_circuit(U.adjoint())

        coeffs = qc_res.get_coeff_vec()
        residuals = []

        # each operator needs a score, so loop over toperators
        for m in self._tops:
            if self._optimizer.lower() == "jacobi":
                # In this case, the sign associated with the projection on K |Phi> = (-1)^p |Phi_K>
                # needs to be taken into consideration
                sq_op = self._pool_obj[m][1]
                # occ => i,j,k,...
                # vir => a,b,c,...
                # sq_op is 1.0(a^ b^ i j) - 1.0(j^ i^ b a)

                qc_temp = qforte.Computer(self._nqb)
                qc_temp.apply_circuit(self._refprep)
                qc_temp.apply_operator(sq_op.jw_transform(self._qubit_excitations))
                sign_adjust = qc_temp.get_coeff_vec()[self._pool_idx_to_coeff_idx[m]]

                res_m = coeffs[self._pool_idx_to_coeff_idx[m]] * sign_adjust
            else:
                # In residual minimization, we compute the function sum_k |r_k|^2
                # and thus the sign of the projection is immaterial
                res_m = coeffs[self._pool_idx_to_coeff_idx[m]]
            if abs(np.imag(res_m)) > 0.0:
                raise ValueError(
                    "Residual has imaginary component, something went wrong!!"
                )
            residuals.append(res_m)

        self._res_vec_norm = np.linalg.norm(residuals)
        self._res_vec_evals += 1
        self._res_m_evals += len(trial_amps)

        return residuals

    def update_ansatz(self):
        self._n_pauli_measures_k = 0
        # TODO: Check if this deepcopy is needed. The one argument of energy_feval should be const.
        x0 = copy.deepcopy(self._tamps)
        init_gues_energy = self.energy_feval(x0)

        # do U^dag e^iH U |Phi_o> = |Phi_res>
        U = self.ansatz_circuit()

        qc_res = qf.Computer(self._nqb)
        qc_res.apply_circuit(self._Uprep)
        qc_res.apply_circuit(U)
        qc_res.apply_circuit(self._eiH)
        qc_res.apply_circuit(U.adjoint())

        res_coeffs = qc_res.get_coeff_vec()

        # build different res_sq list using M_omega
        if self._M_omega != "inf":
            res_sq_tmp = [
                np.real(np.conj(res_coeffs[I]) * res_coeffs[I])
                for I in range(len(res_coeffs))
            ]

            # Nmu_lst => [ det1, det2, det3, ... det_M_omega]
            det_lst = np.random.choice(len(res_coeffs), self._M_omega, p=res_sq_tmp)

            print(f"|Co|dt^2 :       {np.amax(res_sq_tmp):12.14f}")
            print(
                f"mu_o :           {np.where(res_sq_tmp == np.amax(res_sq_tmp))[0][0]}"
            )

            No_idx = np.where(res_sq_tmp == np.amax(res_sq_tmp))[0][0]
            print(f"\nNo_idx   {No_idx:4}")

            No = np.count_nonzero(det_lst == No_idx)
            print(f"\nNo       {No:10}")

            res_sq = []
            Nmu_lst = []
            for mu in range(len(res_coeffs)):
                Nmu = np.count_nonzero(det_lst == mu)
                if Nmu > 0:
                    print(
                        f"mu:    {mu:8}      Nmu      {Nmu:10}  r_mu: { Nmu / (self._M_omega):12.14f} "
                    )
                    Nmu_lst.append((Nmu, mu))
                res_sq.append((Nmu / (self._M_omega), mu))

            ## 1. sort
            Nmu_lst.sort()
            res_sq.sort()

            ## 2. set norm
            self._curr_res_sq_norm = sum(rmu_sq[0] for rmu_sq in res_sq[:-1]) / (
                self._dt * self._dt
            )

            ## 3. print stuff
            print("  \n--> Begin selection opt with residual magnitudes:")
            print("  Initial guess energy:          ", round(init_gues_energy, 10))
            print(
                f"  Norm of approximate res vec:  {np.sqrt(self._curr_res_sq_norm):14.12f}"
            )

            ## 4. check conv status (need up update function with if(M_omega != 'inf'))
            if len(Nmu_lst) == 1:
                print("  SPQE converged with M_omega thresh!")
                self._converged = True
                self._final_energy = self._energies[-1]
                self._final_result = self._results[-1]
            else:
                self._converged = False

            ## 5. add new toperator
            if not self._converged:
                if self._verbose:
                    print("\n")
                    print("     op index (Imu)     Number of times measured")
                    print("  -----------------------------------------------")

                for rmu_sq, global_op_idx in Nmu_lst[:-1]:
                    if self._verbose:
                        print(
                            f"  {global_op_idx:10}                  {np.real(rmu_sq):14}"
                        )
                    if global_op_idx not in self._tops:
                        pool_idx = self._coeff_idx_to_pool_idx[global_op_idx]
                        self._tops.insert(0, pool_idx)
                        self._tamps.insert(0, 0.0)
                        operator_rank = len(self._pool_obj[pool_idx][1].terms()[0][1])
                        self._nbody_counts[operator_rank - 1] += 1

                self._n_classical_params_lst.append(len(self._tops))

        else:  # when M_omega == 'inf', proceed with standard SPQE
            res_sq = [
                (np.real(np.conj(res_coeffs[I]) * res_coeffs[I]), I)
                for I in self._indices_of_zeroable_residuals_for_pool
                - {self._pool_idx_to_coeff_idx[i] for i in self._tops}
            ]
            res_sq.sort()
            self._curr_res_sq_norm = sum(rmu_sq[0] for rmu_sq in res_sq) / (
                self._dt * self._dt
            )

            self.conv_status()

            if not self._converged:
                print("\n\n -----> SPQE iteration ", self._spqe_iter, " <-----\n")
                print("  \n--> Begin selection opt with residual magnitudes |r_mu|:")
                print("  Initial guess energy: ", round(init_gues_energy, 10))
                print(
                    f"  Norm of res vec:      {np.sqrt(self._curr_res_sq_norm):14.12f}"
                )

                if self._verbose:
                    print("\n")
                    print("     op index (Imu)           Residual Factor")
                    print("  -----------------------------------------------")
                res_sq_sum = 0.0

                if self._use_cumulative_thresh:
                    # Make a running list of operators. When the sum of res_sq exceeds the target, every operator
                    # from here out is getting added to the ansatz..
                    temp_ops = []
                    for rmu_sq, global_op_idx in res_sq:
                        res_sq_sum += rmu_sq / (self._dt * self._dt)
                        if res_sq_sum > (self._spqe_thresh * self._spqe_thresh):
                            pool_idx = self._coeff_idx_to_pool_idx[global_op_idx]
                            if self._verbose:
                                print(
                                    f"  {pool_idx:10}                  {np.real(rmu_sq):14.12f}"
                                    f"   {self._pool_obj[pool_idx][1].str()}"
                                )
                            if pool_idx not in self._tops:
                                temp_ops.append(pool_idx)
                                operator_rank = len(
                                    self._pool_obj[pool_idx][1].terms()[0][1]
                                )
                                self._nbody_counts[operator_rank - 1] += 1

                    for temp_op in temp_ops[::-1]:
                        self._tops.insert(0, temp_op)
                        self._tamps.insert(0, 0.0)

                else:
                    # Add the single operator with greatest rmu_sq not yet in the ansatz
                    res_sq.reverse()
                    for rmu_sq, global_op_idx in res_sq:
                        pool_idx = self._coeff_idx_to_pool_idx[global_op_idx]
                        print(
                            f"  {pool_idx:10}                  {np.real(rmu_sq)/(self._dt * self._dt):14.12f}"
                        )
                        if pool_idx not in self._tops:
                            print("Adding this operator to ansatz")
                            self._tops.insert(0, pool_idx)
                            self._tamps.insert(0, 0.0)
                            operator_rank = len(
                                self._pool_obj[pool_idx][1].terms()[0][1]
                            )
                            self._nbody_counts[operator_rank - 1] += 1
                            break

                self._n_classical_params_lst.append(len(self._tops))

    def conv_status(self):
        if abs(self._curr_res_sq_norm) < abs(self._spqe_thresh * self._spqe_thresh):
            self._converged = True
            self._stop_macro = True
            print("\n\n\n------------------------------------------------")
            print("SPQE macro-iterations converged!")
            print(f"||r|| = {np.sqrt(self._curr_res_sq_norm):8.6f}")
            print("------------------------------------------------")
        elif self._spqe_iter > self._spqe_maxiter:
            print("\n\n\n------------------------------------------------")
            print("Maximum number of SPQE macro-iterations reached!")
            print(f"Current value of ||r||: {np.sqrt(self._curr_res_sq_norm):8.6f}")
            print("------------------------------------------------")
            self._converged = False
            self._stop_macro = True
        elif len(self._tops) == len(self._pool_obj):
            print("\n\n\n------------------------------------------------")
            print("Operator pool has been drained!")
            print(f"Current value of ||r||: {np.sqrt(self._curr_res_sq_norm):8.6f}")
            print("------------------------------------------------")
            self._converged = True
            self._stop_macro = True


SPQE.jacobi_solver = optimizer.jacobi_solver
SPQE.scipy_solver = optimizer.scipy_solver
SPQE.construct_moment_space = moment_energy_corrections.construct_moment_space
SPQE.compute_moment_energies = moment_energy_corrections.compute_moment_energies
