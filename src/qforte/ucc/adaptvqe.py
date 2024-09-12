"""
ADAPT-VQE classes
====================================
Classes for using the variational quantum eigensolver
with variants of the Adaptive Derivative-Assembled Pseudo-Trotter
(ADAPT) anxatz.
"""

from qforte.abc.uccvqeabc import UCCVQE
from qforte.experiment import *
from qforte.utils.transforms import *
from qforte.utils.state_prep import *
from qforte.utils import moment_energy_corrections
from qforte.maths import optimizer
from qforte.expansions.excited_state_algorithms import *
import numpy as np
from scipy.optimize import minimize


class ADAPTVQE(UCCVQE):
    """A class that encompasses the three components of using the variational
    quantum eigensolver to optimize a parameterized unitary CC like wave function
    comprised of adaptively selected operators. Growing a circuit over many iterations
    the ADAPT-VQE: (1) prepares a quantum state on the quantum computer
    representing the wave function to be simulated, (2) evaluates the energy by
    and gradients, and (3) optimizes the wave function by minimizing the energy.

    In ADAPT-VQE, the unitary ansatz at macro-iteration :math:`k` is defined as

    .. math::
        \\hat{U}_\\mathrm{ADAPT}^{(k)}(\\mathbf{t}) = \\prod_\\nu^{k} e^{ t_\\nu^{(k)} \\hat{\\kappa}_\\nu^{(k)} },

    where the index :math:`\\nu` is likewise a index corresponding to unique operators
    :math:`\\hat{\\kappa}_\\nu` in a pool of operators.
    Note that the parameters :math:`t_\\nu^{(k)}` are re-optimized at each macro-iteration.
    New operators are determined from the pool by computing the energy gradient

    .. math::
        g_\\nu = \\langle \\Psi_\\mathrm{VQE} | [ \\hat{H}, \\hat{\\kappa}_\\nu ] | \\Psi_\\mathrm{VQE} \\rangle,

    with respect to :math:`t_\\nu` of each operator in the pool and selecting
    the operator with the largest gradient magnitude to place at the end of
    the ansatz in the next iteration.

    Attributes
    ----------

    _avqe_thresh : float
        The gradient norm threshold to determine when the ADAPT-VQE
        algorithm has converged.

    _stop_E : float
        An energy at which ADAPT will terminate, regardless of gradient, number of iterations, etc.

    _commutator_pool : [QubitOpPool]
        The list of [H, X] to be measured

    _converged : bool
        Whether or not the ADAPT-VQE has converged according to the gradient-norm
        threshold.

    _curr_grad_norm : float
        The gradient norm for the current iteration of ADAPT-VQE.

    _energies : list
        The optimized energies from each iteration of ADAPT-VQE.

    _ritz_energies : list of lists
        The eigenstates in the subspace of H defined by the current ADAPT ansatz acting on multiple references.

    _final_energy : float
        The final ADAPT-VQE energy value.

    _final_result : Result
        The last result object from the optimizer.

    _grad_norms : list
        The gradient norms from each iteration of ADAPT-VQE.

    _results : list
        The optimizer result objects from each iteration of ADAPT-VQE.
    """

    def run(
        self,
        avqe_thresh=1.0e-2,
        pool_type="sa_SD",
        opt_thresh=1.0e-5,
        opt_maxiter=200,
        adapt_maxiter=20,
        stop_E=None,
        optimizer="BFGS",
        use_analytic_grad=True,
        use_cumulative_thresh=False,
        add_equiv_ops=False,
        tamps=[],
        tops=[],
    ):
        self._avqe_thresh = avqe_thresh
        self._opt_thresh = opt_thresh
        self._adapt_maxiter = adapt_maxiter

        self._opt_maxiter = opt_maxiter
        self._stop_E = stop_E
        self._use_analytic_grad = use_analytic_grad
        self._optimizer = optimizer
        if self._use_analytic_grad and self._optimizer in {
            "nelder-mead",
            "powell",
            "cobyla",
        }:
            print(f"{self._optimizer} optimizer doesn't support analytic grads.")
            self._use_analytic_grad = False
        self._pool_type = pool_type
        self._use_cumulative_thresh = use_cumulative_thresh
        self._add_equiv_ops = add_equiv_ops

        self._results = []
        self._energies = []
        self._diag_energies = []
        self._diag_As = []
        self._grad_norms = []
        self._tops = copy.deepcopy(tops)
        self._tamps = copy.deepcopy(tamps)
        self._commutator_pool = []
        self._converged = 0

        self._n_ham_measurements = 0
        self._n_commut_measurements = 0

        self._n_classical_params = 0
        self._n_cnot = 0
        self._n_cnot_lst = []
        self._n_classical_params_lst = []
        self._n_pauli_trm_measures = 0
        self._n_pauli_trm_measures_lst = []

        self._res_vec_evals = 0
        self._res_m_evals = 0
        self._k_counter = 0

        self._curr_grad_norm = 0.0
        self._prev_energy = self.energy_feval([])

        # Print options banner (should done for all algorithms).
        self.print_options_banner()

        self.fill_pool()

        if self._max_moment_rank:
            print("\nConstructing Moller-Plesset and Epstein-Nesbet denominators")
            self.construct_moment_space()

        if self._verbose:
            print("\n\n-------------------------------------")
            print("   Second Quantized Operator Pool")
            print("-------------------------------------")
            print(self._pool_obj.str())

        avqe_iter = 0

        hit_maxiter = 0
        if adapt_maxiter == 0:
            hit_maxiter = 1
            self._converged = True

        if self._print_summary_file:
            f = open("summary.dat", "w+", buffering=1)
            f.write(
                f"#{'Iter(k)':>8}{'E(k)':>14}{'N(params)':>17}{'N(CNOT)':>18}{'N(measure)':>20}\n"
            )
            f.write(
                "#-------------------------------------------------------------------------------\n"
            )

        if self._is_multi_state:
            if self._state_prep_type != "computer":
                E, A, ops = qforte.excited_state_algorithms.ritz_eigh(
                    self._nqb, self._qb_ham, self.build_Uvqc()
                )
            else:
                H_eff = qforte.build_effective_array(
                    self._qb_ham, self.build_Uvqc()[0], self.get_initial_computer()
                ).real
                E, A = np.linalg.eigh(H_eff)
            self._diag_energies.append(E)
            self._diag_As.append(A)
            cur_string = f"Current Energies {avqe_iter}"
            diag_string = f"Best Energies {avqe_iter}"
            for e in E:
                diag_string += f" {e}"
                cur_string += f" {e}"
            print(cur_string)
            print(diag_string)

        while not self._converged:
            print("\n\n -----> ADAPT-VQE iteration ", avqe_iter, " <-----\n")
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

            if self._is_multi_state:
                if self._state_prep_type != "computer":
                    E, A, ops = qforte.excited_state_algorithms.ritz_eigh(
                        self._nqb, self._qb_ham, self.build_Uvqc()
                    )
                else:
                    H_eff = qforte.build_effective_array(
                        self._qb_ham, self.build_Uvqc()[0], self.get_initial_computer()
                    ).real
                    E, A = np.linalg.eigh(H_eff)

                self._diag_energies.append(E)
                self._diag_As.append(A)

            if self._verbose:
                print(
                    "\ntamplitudes for tops post solve: \n", list(np.real(self._tamps))
                )
                if self._is_multi_state:
                    diag_string = f"Current Energies {avqe_iter + 1}:"
                    for e in E:
                        diag_string += f" {e}"
                    print(diag_string)
                    best_string = f"Best Energies {avqe_iter + 1}"
                    D = np.array(self._diag_energies)
                    for i in range(len(self._ref)):
                        best_string += f" {np.amin(D[:,i])}"
                    print(best_string)

            if self._print_summary_file:
                f.write(
                    f"  {avqe_iter:7}    {self._energies[-1]:+15.9f}    {len(self._tamps):8}        {self._n_cnot_lst[-1]:10}        {sum(self._n_pauli_trm_measures_lst):12}\n"
                )

            avqe_iter += 1

            if avqe_iter > self._adapt_maxiter - 1:
                hit_maxiter = 1
                break

        if self._print_summary_file:
            f.close()

        # Set final ground state energy.
        if hit_maxiter:
            self._Egs = self.get_final_energy(hit_max_avqe_iter=1)
            if self._optimizer.lower() != "jacobi":
                if len(self._results) == 0:
                    self._final_result = None
                else:
                    self._final_result = self._results[-1]

        if self._is_multi_state:
            # Check that ref and final are included
            best_diags = []
            E_arr = np.array(self._diag_energies)
            for i in range(len(self._ref)):
                best_diags.append(np.amin(E_arr[:, i]))
            self._best_diags = best_diags

        self._Egs = self.get_final_energy()

        print("\n\n")
        if not self._max_moment_rank:
            print(
                f"{'Iter':>8}{'E':>14}{'N(params)':>17}{'N(CNOT)':>18}{'N(measure)':>20}"
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

        if len(self._tamps) > 0:
            self._n_classical_params = len(self._tamps)
            self._n_cnot = self._n_cnot_lst[-1]
            self._n_pauli_trm_measures = sum(self._n_pauli_trm_measures_lst)

            # Print summary banner (should done for all algorithms).
            self.print_summary_banner()

            # verify that required attributes were defined
            # (should be called for all algorithms!)
            self.verify_run()

    # Define Algorithm abstract methods.
    def run_realistic(self):
        raise NotImplementedError(
            "run_realistic() is not fully implemented for ADAPT-VQE."
        )

    def verify_run(self):
        self.verify_required_attributes()
        self.verify_required_VQE_attributes()
        self.verify_required_UCCVQE_attributes()

    def print_options_banner(self):
        print("\n-----------------------------------------------------")
        print("  Adaptive Derivative-Assembled Pseudo-Trotter VQE   ")
        print("-----------------------------------------------------")

        print("\n\n               ==> ADAPT-VQE options <==")
        print("---------------------------------------------------------")

        self.print_generic_options()

        print("Use qubit excitations:                   ", self._qubit_excitations)
        print("Use compact excitation circuits:         ", self._compact_excitations)

        # VQE options.
        opt_thrsh_str = "{:.2e}".format(self._opt_thresh)
        avqe_thrsh_str = "{:.2e}".format(self._avqe_thresh)
        print("Optimization algorithm:                  ", self._optimizer)
        print("Optimization maxiter:                    ", self._opt_maxiter)
        print("Optimizer grad-norm threshold (theta):   ", opt_thrsh_str)

        # UCCVQE options.
        print("Use analytic gradient:                   ", str(self._use_analytic_grad))
        print("Operator pool type:                      ", str(self._pool_type))

        # Specific ADAPT-VQE options.
        print("ADAPT-VQE grad-norm threshold (eps):     ", avqe_thrsh_str)
        print("ADAPT-VQE maxiter:                       ", self._adapt_maxiter)

    def print_summary_banner(self):
        print("\n\n                ==> ADAPT-VQE summary <==")
        print("-----------------------------------------------------------")
        print("Final ADAPT-VQE Energy:                     ", round(self._Egs, 10))
        if self._max_moment_rank:
            print(
                "Moment-corrected (MP) ADAPT-VQE Energy:     ",
                round(self._E_mmcc_mp[-1], 10),
            )
            print(
                "Moment-corrected (EN) ADAPT-VQE Energy:     ",
                round(self._E_mmcc_en[-1], 10),
            )
        print("Number of operators in pool:                 ", len(self._pool_obj))
        print("Final number of amplitudes in ansatz:        ", len(self._tamps))
        print(
            "Total number of Hamiltonian measurements:    ",
            self.get_num_ham_measurements(),
        )
        print(
            "Total number of commutator measurements:      ",
            self.get_num_commut_measurements(),
        )
        print("Number of classical parameters used:         ", self._n_classical_params)
        print("Number of CNOT gates in deepest circuit:     ", self._n_cnot)
        print(
            "Number of Pauli term measurements:           ", self._n_pauli_trm_measures
        )

        print("Number of grad vector evaluations:           ", self._res_vec_evals)
        print("Number of individual grad evaluations:       ", self._res_m_evals)

    # Define VQE abstract methods.
    def solve(self):
        if self._optimizer.lower() == "jacobi":
            self.build_orb_energies()
            return self.jacobi_solver()
        else:
            return self.scipy_solve()

    def scipy_solve(self):
        self._k_counter = 0

        opts = {}
        opts["gtol"] = self._opt_thresh
        opts["disp"] = False
        opts["maxiter"] = self._opt_maxiter
        x0 = copy.deepcopy(self._tamps)

        init_gues_energy = self.energy_feval(x0)

        self._prev_energy = init_gues_energy
        if not self._is_multi_state:
            factor = 1
        else:
            factor = len(self._ref)
        if self._use_analytic_grad:
            print("  \n--> Begin opt with analytic gradient:")
            print(f" Initial guess energy:              {init_gues_energy:+12.10f}")
            res = minimize(
                self.energy_feval,
                x0,
                method=self._optimizer,
                jac=self.gradient_ary_feval,
                options=opts,
                callback=self.report_iteration,
            )

            # account for energy evaluations
            self._n_pauli_measures_k += factor * self._Nl * res.nfev

            # account for gradient evaluations
            for m in self._tops:
                self._n_pauli_measures_k += factor * self._Nm[m] * self._Nl * res.njev

        else:
            print("  \n--> Begin opt with grad estimated using first-differences:")
            print(f" Initial guess energy:              {init_gues_energy:+12.10f}")
            res = minimize(
                self.energy_feval,
                x0,
                method=self._optimizer,
                options=opts,
                callback=self.report_iteration,
            )

            self._n_pauli_measures_k += factor * self._Nl * res.nfev

        if res.success:
            print("  => Minimization successful!")
            print(f"  => Minimum Energy: {res.fun:+12.10f}")

        else:
            print("  => WARNING: minimization result may not be tightly converged.")
            print(f"  => Minimum Energy: {res.fun:+12.10f}")

        if res.x[-1] == 0:
            if len(self._energies) == 0:
                print("ADAPT Already Converged.")
                self._tops = []
                self._tamps = []
                self._converged = True
                self._final_energy = []
                self._final_results = []
            else:
                print(
                    "  => WARNING: ADAPT could not optimize the new parameter.  Deleting new parameter and terminating the algorithm."
                )
                print(f"  => Minimum Energy: {self._energies[-1]:+12.10f}")
                self._converged = True
                self._final_energy = self._energies[-1]
                self._final_results = self._results[-1]
                self._tops = self._tops[:-1]
                self._tamps = self._tamps[:-1]
        else:
            self._energies.append(res.fun)
            self._results.append(res)
            self._tamps = list(res.x)
            self._n_pauli_trm_measures_lst.append(self._n_pauli_measures_k)
        if not self._is_multi_state:
            self._n_cnot_lst.append(self.build_Uvqc().get_num_cnots())
        else:
            U_temp = self.build_Uvqc()
            cnots = [U.get_num_cnots() for U in U_temp]
            self._n_cnot_lst.append(max(cnots))
            del U_temp
            del cnots

    # Define ADAPT-VQE methods.
    def update_ansatz(self):
        """Adds a parameter and operator to the ADAPT-VQE circuit based on the
        magnitude of the gradients of pool operators, checks for
        convergence.
        """
        self._n_pauli_measures_k = 0

        curr_norm = 0.0
        lgrst_grad = 0.0

        if self._verbose:
            print(
                "     op index (m)     N pauli terms              Gradient            Tmu  "
            )
            print(
                "  ------------------------------------------------------------------------------"
            )

        grads = self.measure_gradient3()

        for m, grad_m in enumerate(grads):
            if not self._is_multi_state:
                factor = 1
            else:
                factor = len(self._ref)
            # refers to number of times sigma_y must be measured in "strategies for UCC" grad eval circuit
            self._n_pauli_measures_k += factor * self._Nl * self._Nm[m]

            curr_norm += grad_m**2
            if self._verbose:
                print(
                    f"       {m:3}                {self._Nm[m]:8}             {grad_m:+12.9f}      {self._pool_obj[m][1].terms()[0][1]}"
                )

            if abs(grad_m) > abs(lgrst_grad):
                if abs(lgrst_grad) > 0.0:
                    secnd_lgst_grad = lgrst_grad
                    secnd_lgrst_grad_idx = lgrst_grad_idx

                lgrst_grad = grad_m
                lgrst_grad_idx = m

        curr_norm = np.sqrt(curr_norm)
        print("\n==> Measuring gradients from pool:")
        print(" Norm of <[H,Am]> = %12.8f" % curr_norm)
        print(" Max  of <[H,Am]> = %12.8f" % lgrst_grad)

        self._curr_grad_norm = curr_norm
        self._grad_norms.append(curr_norm)

        self.conv_status()

        if not self._converged:
            if self._use_cumulative_thresh:
                temp_order_tops = []
                grads_sq = [(grads[m] * grads[m], m) for m in range(len(grads))]
                grads_sq.sort()
                gm_sq_sum = 0.0
                for m, gm_sq in enumerate(grads_sq):
                    gm_sq_sum += gm_sq[0]
                    if gm_sq_sum > (self._avqe_thresh * self._avqe_thresh):
                        print(
                            f"  Adding operator m =     {gm_sq[1]:10}   |gm| = {np.sqrt(gm_sq[0]):10.8f}"
                        )
                        self._tamps.append(0.0)
                        temp_order_tops.insert(0, gm_sq[1])

                self._tops.extend(copy.deepcopy(temp_order_tops))

            else:
                print("  Adding operator m =", lgrst_grad_idx)

                if (
                    len(self._tops) > 0
                    and self._stop_E != None
                    and self._energies[-1] < self._stop_E
                ):
                    print("Energy is below the targeted accuracy.")
                    self._converged = True
                    self._final_energy = self._energies[-1]
                    if self._optimizer.lower() != "jacobi":
                        self._final_result = self._results[-1]

                elif len(self._tops) > 0 and self._tops[-1] == lgrst_grad_idx:
                    print(
                        "ADAPT wants to add the same operator as the previous iteration.  Aborting."
                    )
                    self._converged = True
                    self._final_energy = self._energies[-1]
                    if self._optimizer.lower() != "jacobi":
                        self._final_result = self._results[-1]

                else:
                    self._tops.append(lgrst_grad_idx)
                    self._tamps.append(0.0)

                    if self._add_equiv_ops:
                        if abs(lgrst_grad) - abs(secnd_lgst_grad) < 1.0e-5:
                            print(" *Adding operator m =", secnd_lgrst_grad_idx)
                            self._tops.append(secnd_lgrst_grad_idx)
                            self._tamps.append(0.0)

            self._n_classical_params_lst.append(len(self._tops))

        else:
            print("\n  ADAPT-VQE converged!")

    def conv_status(self):
        """Sets the convergence states."""
        if abs(self._curr_grad_norm) < abs(self._avqe_thresh):
            self._converged = True
            self._final_energy = self._energies[-1]
            if self._optimizer.lower() != "jacobi":
                self._final_result = self._results[-1]
        else:
            self._converged = False
        print("", flush=True)

    def get_num_ham_measurements(self):
        if not self._is_multi_state:
            factor = 1
        else:
            factor = len(self._ref)

        for res in self._results:
            self._n_ham_measurements += factor * res.nfev
        return self._n_ham_measurements

    def get_num_commut_measurements(self):
        if not self._is_multi_state:
            factor = 1
        else:
            factor = len(self._ref)

        self._n_commut_measurements += factor * len(self._tamps) * len(self._pool_obj)

        if self._use_analytic_grad:
            for m, res in enumerate(self._results):
                self._n_commut_measurements += factor * res.njev * (m + 1)

        return self._n_commut_measurements

    def get_final_energy(self, hit_max_avqe_iter=False):
        """
        Parameters
        ----------
        hit_max_avqe_iter : bool
            Whether ADAPT-VQE has already hit the maximum number of iterations.
        """
        if hit_max_avqe_iter:
            if len(self._energies) == 0:
                print("\nADAPT-VQE did no iterations as requested.")
                self._final_energy = None
            else:
                print("\nADAPT-VQE at maximum number of iterations.")
                self._final_energy = self._energies[-1]
        else:
            return self._final_energy

    def get_final_result(self, hit_max_avqe_iter=False):
        """
        Parameters
        ----------
        hit_max_avqe_iter : bool
            Whether ADAPT-VQE has already hit the maximum number of iterations.
        """
        if hit_max_avqe_iter:
            if len(self._results) == 0:
                print("\nADAPT-VQE did no iterations as requested.")
                self._final_result = None
            else:
                print("\nADAPT-VQE at maximum number of iterations.")
                self._final_result = self._results[-1]
        else:
            return self._final_result


ADAPTVQE.jacobi_solver = optimizer.jacobi_solver
ADAPTVQE.construct_moment_space = moment_energy_corrections.construct_moment_space
ADAPTVQE.compute_moment_energies = moment_energy_corrections.compute_moment_energies
