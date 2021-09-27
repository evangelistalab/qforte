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

import numpy as np

class SPQE(UCCPQE):
    """This class implements the selected projective quantum eigensolver (SPQE) for
    disentangled UCC like ansatz.
    In SPQE, a batch of important particle-hole operators
    :math:`\{ e^{t_\mu (\hat{\\tau}_\mu - \hat{\\tau}_\mu^\dagger )} \}` are
    added at each macro-iteration :math:`n` to the SPQE unitary :math:`\hat{U}(\mathbf{t})`,
    wile all current parameters are optemized using the quasi-Newton PQE update
    with micro-iterations :math:`k`.

    In our selection approach we consider a (normalized) quantum state of the form

    .. math::
        | \\tilde{r} \\rangle  = \\tilde{r}_0 | \Phi_0 \\rangle + \sum_\mu \\tilde{r}_\mu  | \Phi_\mu \\rangle

    where the quantities :math:`\\tilde{r}_\mu` are approximately proportional to
    the residuals :math:`r_\mu`.
    The state :math:`| \\tilde{r} \\rangle` can be approximately reproduced via

    .. math::
        | \\tilde{r} \\rangle \\approx \hat{U}^\dagger e^{i \Delta t \hat{H}} \hat{U} | \Phi_0 \\rangle

    .. math::
        \\approx (1 + i\Delta t \hat{U}^\dagger \hat{H} \hat{U})  | \Phi_0 \\rangle + \mathcal{O}(\Delta t^2).

    We note that in this implementation we use a Trotter approximation for the time
    evolution unitary.
    Measuring :math:`\\langle \hat{Z} \\rangle` for each qubit yields a bitstring
    that has corresponding determinat and operator
    :math:`(\hat{\\tau}_\mu - \hat{\\tau}_\mu^\dagger )`
    with probablility proportional to :math:`|\\tilde{r}_\mu|^2`.
    The operators corresponding to the largest :math:`|\\tilde{r}_\mu|^2` values
    are then added to :math:`\hat{U}(\mathbf{t})` at each macro-iteration.
    """
    def run(self,
            spqe_thresh=1.0e-2,
            spqe_maxiter=20,
            dt=0.001,
            M_omega = 'inf',
            opt_thresh = 1.0e-5,
            opt_maxiter = 30,
            use_cumulative_thresh=True):

        if(self._state_prep_type != 'occupation_list'):
            raise ValueError("SPQE implementation can only handle occupation_list Hartree-Fock reference.")

        self._spqe_thresh = spqe_thresh
        self._spqe_maxiter = spqe_maxiter
        self._dt = dt
        if(M_omega != 'inf'):
            self._M_omega = int(M_omega)
        else:
            self._M_omega = M_omega

        self._use_cumulative_thresh = use_cumulative_thresh
        self._opt_thresh = opt_thresh
        self._opt_maxiter = opt_maxiter

        self._nbody_counts = []
        self._n_classical_params_lst = []

        self._results = []
        self._energies = []
        self._grad_norms = []
        self._tops = []
        self._tamps = []
        self._converged = False
        self._res_vec_evals = 0
        self._res_m_evals = 0
        # list: tuple(excited determinant, phase_factor)
        self._excited_dets = []

        self._curr_energy = 0.0

        self._n_classical_params = 0
        self._n_cnot = 0
        self._n_cnot_lst = []
        self._n_pauli_trm_measures = 0
        self._n_pauli_trm_measures_lst = []

        self.print_options_banner()

        self._Nm = []
        self._pool_type = 'full'
        self._eiH, self._eiH_phase = trotterize(self._qb_ham, factor= self._dt*(0.0 + 1.0j), trotter_number=self._trotter_number)

        for occupation in self._ref:
            if occupation:
                self._nbody_counts.append(0)

        pool_obj = qf.SQOpPool()
        for I in range(2 ** self._nqb):
            pool_obj.add_term(0.0, self.get_op_from_basis_idx(I))
        self._qubit_pool = pool_obj.get_qubit_op_pool()

        self.build_orb_energies()
        spqe_iter = 0
        hit_maxiter = 0

        if(self._print_summary_file):
            f = open("summary.dat", "w+", buffering=1)
            f.write(f"#{'Iter(k)':>8}{'E(k)':>14}{'N(params)':>17}{'N(CNOT)':>18}{'N(measure)':>20}\n")
            f.write('#-------------------------------------------------------------------------------\n')

        while not self._converged:

            print('\n\n -----> SPQE iteration ', spqe_iter, ' <-----\n')
            self.update_ansatz()

            if self._converged:
                break

            if(self._verbose):
                print('\ntoperators included from pool: \n', self._tops)
                print('\ntamplitudes for tops: \n', self._tamps)

            self.solve()

            if(self._verbose):
                print('\ntamplitudes for tops post solve: \n', np.real(self._tamps))

            if(self._print_summary_file):
                f.write(f'  {spqe_iter:7}    {self._energies[-1]:+15.9f}    {len(self._tamps):8}        {self._n_cnot_lst[-1]:10}        {sum(self._n_pauli_trm_measures_lst):12}\n')
            spqe_iter += 1

            if spqe_iter > self._spqe_maxiter-1:
                hit_maxiter = 1
                break

        if(self._print_summary_file):
            f.close()

        if hit_maxiter:
            self._Egs = self.get_final_energy(hit_max_spqe_iter=1)

        self._Egs = self.get_final_energy()

        print("\n\n")
        print("---> Final n-body excitation counts in SPQE ansatz <---")
        print("\n")
        print(f"{'Excitaion order':>20}{'Number of operators':>30}")
        print('---------------------------------------------------------')
        for l, nl in enumerate(self._nbody_counts):
            print(f"{l+1:12}              {nl:14}")

        print('\n\n')
        print(f"{'Iter(k)':>8}{'E(k)':>14}{'N(params)':>17}{'N(CNOT)':>18}{'N(measure)':>20}")
        print('-------------------------------------------------------------------------------')

        for k, Ek in enumerate(self._energies):
            print(f' {k:7}    {Ek:+15.9f}    {self._n_classical_params_lst[k]:8}        {self._n_cnot_lst[k]:10}        {sum(self._n_pauli_trm_measures_lst[:k+1]):12}')

        self._n_classical_params = len(self._tamps)
        self._n_cnot = self._n_cnot_lst[-1]
        self._n_pauli_trm_measures = sum(self._n_pauli_trm_measures_lst)

        self.print_summary_banner()
        self.verify_run()

    def run_realistic(self):
        raise NotImplementedError('run_realistic() is not fully implemented for SPQE.')

    def verify_run(self):
        self.verify_required_attributes()
        self.verify_required_PQE_attributes()
        self.verify_required_UCCPQE_attributes()

    def print_options_banner(self):
        print('\n-----------------------------------------------------')
        print('       Selected Projective Quantum Eigensolver   ')
        print('-----------------------------------------------------')

        print('\n\n               ==> SPQE options <==')
        print('---------------------------------------------------------')
        print('Trial reference state:                   ',  ref_string(self._ref, self._nqb))
        print('Number of Hamiltonian Pauli terms:       ',  self._Nl)
        print('Trial state preparation method:          ',  self._state_prep_type)
        print('Trotter order (rho):                     ',  self._trotter_order)
        print('Trotter number (m):                      ',  self._trotter_number)
        print('Use fast version of algorithm:           ',  str(self._fast))
        if(self._fast):
            print('Measurement varience thresh:             ',  'NA')
        else:
            print('Measurement varience thresh:             ',  0.01)

        opt_thrsh_str = '{:.2e}'.format(self._opt_thresh)
        spqe_thrsh_str = '{:.2e}'.format(self._spqe_thresh)
        print('DIIS maxiter:                            ',  self._opt_maxiter)
        print('DIIS residual-norm threshold (omega_r):  ',  opt_thrsh_str)
        print('Operator pool type:                      ',  'full')
        print('SPQE residual-norm threshold (Omega):    ',  spqe_thrsh_str)
        print('SPQE maxiter:                            ',  self._spqe_maxiter)


    def print_summary_banner(self):
        print('\n\n                ==> SPQE summary <==')
        print('-----------------------------------------------------------')
        print('Final SPQE Energy:                           ', round(self._Egs, 10))
        print('Number of operators in pool:                 ', len(self._qubit_pool))
        print('Final number of amplitudes in ansatz:        ', len(self._tamps))
        print('Number of classical parameters used:         ', self._n_classical_params)
        print('Number of CNOT gates in deepest circuit:     ', self._n_cnot)
        print('Number of Pauli term measurements:           ', self._n_pauli_trm_measures)
        print('Number of residual vector evaluations:       ', self._res_vec_evals)
        print('Number of individual residual evaluations:   ', self._res_m_evals)

    def solve(self):
        self.diis_solve()

    def diis_solve(self):
        # draws heavy insiration from Daniel Smith's ccsd_diss.py code in psi4 numpy
        diis_dim = 0
        t_diis = [copy.deepcopy(self._tamps)]
        e_diis = []
        rk_norm = 1.0
        Ek0 = self.energy_feval(self._tamps)

        print('\n    k iteration         Energy               dE           Nrvec ev      Nrm ev*         ||r||')
        print('---------------------------------------------------------------------------------------------------')

        for k in range(1, self._opt_maxiter+1):
            t_old = copy.deepcopy(self._tamps)

            #do regular update
            r_k = self.get_residual_vector(self._tamps)
            rk_norm = np.linalg.norm(r_k)
            r_k = self.get_res_over_mpdenom(r_k)

            self._tamps = list(np.add(self._tamps, r_k))

            Ek = self.energy_feval(self._tamps)
            dE = Ek - Ek0
            Ek0 = Ek
            # self._num_res_evals += 1
            self._res_vec_evals += 1
            self._res_m_evals += len(self._tamps)

            print(f'     {k:7}        {Ek:+12.10f}      {dE:+12.10f}      {self._res_vec_evals:4}        {self._res_m_evals:6}       {rk_norm:+12.10f}')

            if(rk_norm < self._opt_thresh):
                self._results.append('Fake result string')
                self._final_result = 'nothing'
                self._Egs = Ek
                break

            t_diis.append(copy.deepcopy(self._tamps))
            e_diis.append(np.subtract(copy.deepcopy(self._tamps), t_old))

            if(k >= 1):
                diis_dim = len(t_diis) - 1

                #consturct diis B matrix (following Crawford Group github tutorial)
                B = np.ones((diis_dim+1, diis_dim+1)) * -1
                bsol = np.zeros(diis_dim+1)
                B[-1, -1] = 0.0
                bsol[-1] = -1.0

                for i in range(len(e_diis)):
                    for j in range(i, len(e_diis)):
                        B[i,j] = np.dot(np.real(e_diis[i]), np.real(e_diis[j]))
                        if(i!=j):
                            B[j,i] = B[i,j]

                B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()
                x = np.linalg.solve(B, bsol)

                t_new = np.zeros(( len(self._tamps) ))
                for l in range(diis_dim):
                    temp_ary = x[l] * np.asarray(t_diis[l+1])
                    t_new = np.add(t_new, temp_ary)

                self._tamps = copy.deepcopy(list(np.real(t_new)))

        self._Egs = Ek
        self._energies.append(Ek)
        self._n_pauli_measures_k += self._Nl*k * (2*len(self._tamps) + 1)
        self._n_pauli_trm_measures_lst.append(self._n_pauli_measures_k)
        self._n_cnot_lst.append(self.build_Uvqc().get_num_cnots())


    def get_residual_vector(self, trial_amps):
        U = self.ansatz_circuit(trial_amps)

        qc_res = qforte.Computer(self._nqb)
        qc_res.apply_circuit(self._Uprep)
        qc_res.apply_circuit(U)
        qc_res.apply_operator(self._qb_ham)
        qc_res.apply_circuit(U.adjoint())

        coeffs = qc_res.get_coeff_vec()
        residuals = []

        for I, phase in self._excited_dets:

            res_m = coeffs[I] * phase
            if(np.imag(res_m) > 0.0):
                raise ValueError("residual has imaginary component, something went wrong!!")

            residuals.append(res_m)

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
        if(self._M_omega != 'inf'):
            res_sq_tmp = [ np.real(np.conj(res_coeffs[I]) * res_coeffs[I]) for I in range(len(res_coeffs))]

            # Nmu_lst => [ det1, det2, det3, ... det_M_omega]
            det_lst = np.random.choice(len(res_coeffs), self._M_omega, p=res_sq_tmp)

            print(f'|Co|dt^2 :       {np.amax(res_sq_tmp):12.14f}')
            print(f'mu_o :           {np.where(res_sq_tmp == np.amax(res_sq_tmp))[0][0]}')

            No_idx = np.where(res_sq_tmp == np.amax(res_sq_tmp))[0][0]
            print(f'\nNo_idx   {No_idx:4}')

            No = np.count_nonzero(det_lst == No_idx)
            print(f'\nNo       {No:10}')

            res_sq = []
            Nmu_lst = []
            for mu in range(len(res_coeffs)):
                Nmu = np.count_nonzero(det_lst == mu)
                if(Nmu > 0):
                    print(f'mu:    {mu:8}      Nmu      {Nmu:10}  r_mu: { Nmu / (self._M_omega):12.14f} ')
                    Nmu_lst.append((Nmu, mu))
                res_sq.append( ( Nmu / (self._M_omega), mu) )

            ## 1. sort
            Nmu_lst.sort()
            res_sq.sort()

            ## 2. set norm
            self._curr_res_sq_norm = sum(rmu_sq[0] for rmu_sq in res_sq[:-1]) / (self._dt * self._dt)

            ## 3. print stuff
            print('  \n--> Begin selection opt with residual magnitudes:')
            print('  Initial guess energy:          ', round(init_gues_energy,10))
            print(f'  Norm of approximate res vec:  {np.sqrt(self._curr_res_sq_norm):14.12f}')

            ## 4. check conv status (need up update function with if(M_omega != 'inf'))
            if(len(Nmu_lst)==1):
                print('  SPQE converged with M_omega thresh!')
                self._converged = True
                self._final_energy = self._energies[-1]
                self._final_result = self._results[-1]
            else:
                self._converged = False

            ## 5. add new toperator
            if not self._converged:
                if self._verbose:
                    print('\n')
                    print('     op index (Imu)     Number of times measured')
                    print('  -----------------------------------------------')

                for Nmu_tup in Nmu_lst[:-1]:
                    if(self._verbose):
                        print(f"  {Nmu_tup[1]:10}                  {np.real(Nmu_tup[0]):14}")
                    if(Nmu_tup[1] not in self._tops):
                        self.add_index_to_pool(Nmu_tup[1])

                self._n_classical_params_lst.append(len(self._tops))

        else: # when M_omega == 'inf', proceed with standard SPQE
            res_sq = [( np.real(np.conj(res_coeffs[I]) * res_coeffs[I]), I) for I in range(len(res_coeffs))]
            res_sq.sort()
            self._curr_res_sq_norm = sum(rmu_sq[0] for rmu_sq in res_sq[:-1]) / (self._dt * self._dt)

            print('  \n--> Begin selection opt with residual magnitudes |r_mu|:')
            print('  Initial guess energy: ', round(init_gues_energy,10))
            print(f'  Norm of res vec:      {np.sqrt(self._curr_res_sq_norm):14.12f}')

            self.conv_status()

            if not self._converged:
                if self._verbose:
                    print('\n')
                    print('     op index (Imu)           Residual Factor')
                    print('  -----------------------------------------------')
                res_sq_sum = 0.0

                reference_state = qforte.QubitBasis(self._nqb)
                for k, occ in enumerate(self._ref):
                    reference_state.set_bit(k, occ)

                if(self._use_cumulative_thresh):
                    # Make a running list of operators. When the sum of res_sq exceeds the target, every operator
                    # from here out is getting added to the ansatz..
                    op_indices = []
                    for rmu_sq, op_idx in res_sq[:-1]:
                        res_sq_sum += rmu_sq / (self._dt * self._dt)
                        if res_sq_sum > (self._spqe_thresh * self._spqe_thresh):
                            if(self._verbose):
                                Ktemp = self.get_op_from_basis_idx(op_idx)
                                print(f"  {op_idx:10}                  {np.real(rmu_sq)/(self._dt * self._dt):14.12f}   {Ktemp.str()}" )
                            if op_idx not in self._tops:
                                op_indices.append(op_idx)

                    for op_idx in op_indices[::-1]:
                        self.add_index_to_pool(op_idx)

                else:
                    # Add the single operator with greatest rmu_sq not yet in the ansatz
                    res_sq.reverse()
                    for rmu_sq, op_idx in res_sq[1:]:
                        print(f"  {op_idx:10}                  {np.real(rmu_sq)/(self._dt * self._dt):14.12f}")
                        if op_idx not in self._tops:
                            print('Adding this operator to ansatz')
                            self.add_index_to_pool(op_idx)
                            break

                self._n_classical_params_lst.append(len(self._tops))

    def add_from_basis_idx(self, I):

        max_nbody = len(self._nbody_counts)
        nqb = len(self._ref)
        nel = int(sum(self._ref))

        # TODO(Nick): incorparate more flexability into this
        na_el = int(nel/2);
        nb_el = int(nel/2);
        basis_I = qf.QubitBasis(I)

        nbody = 0
        pn = 0
        na_I = 0
        nb_I = 0
        holes = [] # i, j, k, ...
        particles = [] # a, b, c, ...
        parity = []

        # for ( p=0; p<nel; p++) {
        for p in range(nel):
            bit_val = int(basis_I.get_bit(p))
            nbody += (1 - bit_val)
            pn += bit_val
            if(p%2==0):
                na_I += bit_val
            else:
                nb_I += bit_val

            if(bit_val-1):
                holes.append(p)
                if(p%2==0):
                    parity.append(1)
                else:
                    parity.append(-1)

        for q in range(nel, nqb):
            bit_val = int(basis_I.get_bit(q))
            pn += bit_val
            if(q%2==0):
                na_I += bit_val
            else:
                nb_I += bit_val

            if(bit_val):
                particles.append(q)
                if(q%2==0):
                    parity.append(1)
                else:
                    parity.append(-1)

        if(pn==nel and na_I == na_el and nb_I == nb_el):
            if (nbody != 0 and nbody <= max_nbody ):

                total_parity = 1
                for z in parity:
                    total_parity *= z

                if(total_parity==1):
                    K_temp = qf.SQOperator()
                    K_temp.add(+1.0, particles, holes);
                    K_temp.add(-1.0, holes[::-1], particles[::-1]);
                    K_temp.simplify();
                    # this is potentially slow
                    self._Nm.insert(0, len(K_temp.jw_transform().terms()))
                    self._nbody_counts[nbody-1] += 1

    def get_op_from_basis_idx(self, I):

        max_nbody = len(self._nbody_counts)
        nqb = len(self._ref)
        nel = int(sum(self._ref))

        # TODO(Nick): incorparate more flexability into this
        na_el = int(nel/2);
        nb_el = int(nel/2);
        basis_I = qf.QubitBasis(I)

        nbody = 0
        pn = 0
        na_I = 0
        nb_I = 0
        holes = [] # i, j, k, ...
        particles = [] # a, b, c, ...
        parity = []

        for p in range(nel):
            bit_val = int(basis_I.get_bit(p))
            nbody += (1 - bit_val)
            pn += bit_val
            if(p%2==0):
                na_I += bit_val
            else:
                nb_I += bit_val

            if(bit_val-1):
                holes.append(p)
                if(p%2==0):
                    parity.append(1)
                else:
                    parity.append(-1)

        for q in range(nel, nqb):
            bit_val = int(basis_I.get_bit(q))
            pn += bit_val
            if(q%2==0):
                na_I += bit_val
            else:
                nb_I += bit_val

            if(bit_val):
                particles.append(q)
                if(q%2==0):
                    parity.append(1)
                else:
                    parity.append(-1)

        if(pn==nel and na_I == na_el and nb_I == nb_el):
            if (nbody==0):
                return qf.SQOperator()
            if (nbody != 0 and nbody <= max_nbody ):

                total_parity = 1
                for z in parity:
                    total_parity *= z

                if(total_parity==1):
                    K_temp = qf.SQOperator()
                    K_temp.add(+1.0, particles, holes);
                    K_temp.add(-1.0, holes[::-1], particles[::-1]);
                    K_temp.simplify();

                    return K_temp

        return qf.SQOperator()

    def conv_status(self):
        if abs(self._curr_res_sq_norm) < abs(self._spqe_thresh * self._spqe_thresh):
            self._converged = True
            self._final_energy = self._energies[-1]
            self._final_result = self._results[-1]
        else:
            self._converged = False

    def add_index_to_pool(self, index):
        reference_state = qforte.QubitBasis(self._nqb)
        for k, occ in enumerate(self._ref):
            reference_state.set_bit(k, occ)

        self._tops.insert(0, index)
        self._tamps.insert(0, 0.0)
        self.add_from_basis_idx(index)
        self._excited_dets.insert(0, operator_to_determinant(self._qubit_pool[index][1], reference_state))

    def get_final_energy(self, hit_max_spqe_iter=0):
        """
        Parameters
        ----------
        hit_max_spqe_iter : bool
            Wether or not to use the SPQE has already hit the maximum
            number of iterations.
        """
        if hit_max_spqe_iter:
            print("\nSPQE at maximum number of iterations!")
            self._final_energy = self._energies[-1]
        else:
            return self._final_energy
