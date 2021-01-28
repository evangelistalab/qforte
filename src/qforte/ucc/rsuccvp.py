"""
rsuccvp.py
====================================
A class for...
"""

import qforte as qf

from qforte.abc.uccvqeabc import UCCVQE
from qforte.experiment import *
from qforte.utils.transforms import *
from qforte.utils.op_pools import *
from qforte.utils.state_prep import *
from qforte.utils.trotterization import trotterize

import numpy as np
from scipy.optimize import minimize

class RSUCC(UCCVQE):
    """

    """
    def run(self,
            avqe_thresh=1.0e-2,
            rsucc_thresh=1.0e-2,
            pool_type='SD',
            succ_maxiter=20,
            use_analytic_grad = True,
            op_select_type='total_res',
            dt=0.01,
            use_adaptive_t = False,
            M_omega = 'inf',
            res_vec_thresh = 1.0e-5,
            max_residual_iter = 30,
            use_cumulative_thresh=False,
            use_comutator_grad_selection=False):

        self._rsucc_thresh = rsucc_thresh
        self._succ_maxiter = succ_maxiter
        self._use_analytic_grad = use_analytic_grad

        self._optimizer = 'ucc eqations'        # remove this with new base class
        self._opt_maxiter = max_residual_iter   # remove this with new base class
        self._opt_thresh = res_vec_thresh

        self._pool_type = pool_type
        self._op_select_type = op_select_type
        self._dt = dt
        self._use_adaptive_t = use_adaptive_t
        if(M_omega != 'inf'):
            self._M_omega = int(M_omega)
        else:
            self._M_omega = M_omega

        self._use_cumulative_thresh = use_cumulative_thresh
        self._use_comutator_grad_selection = use_comutator_grad_selection

        self._res_vec_thresh = res_vec_thresh
        self._max_residual_iter = max_residual_iter

        self._nbody_counts = []

        self._n_classical_params_lst = []

        self._results = []
        self._energies = []
        self._grad_norms = []
        self._tops = []
        self._tamps = []
        self._comutator_pool = []
        self._converged = 0

        self._num_res_evals = 0

        self._res_vec_evals = 0
        self._res_m_evals = 0

        self._prev_energy = 0.0
        self._curr_energy = 0.0

        self._n_ham_measurements = 0
        self._n_comut_measurements = 0

        self._n_classical_params = 0
        self._n_cnot = 0
        self._n_cnot_lst = []
        self._n_pauli_trm_measures = 0
        self._n_pauli_trm_measures_lst = []

        # Print options banner (should done for all algorithms).
        self.print_options_banner()

        if(op_select_type=='total_res'):
            print('\nUsing total residual selection, no pre defined pool')
            print('Nm does not give number of terms in pauli op')
            self._pool = [0.0 for I in range(2**self._nqb)]

            self._pool_obj = qf.SQOpPool()
            self._grad_vec_evals = 0
            self._Nm = []
            self._pool_type = 'complete'
            self._eiH, self._eiH_phase = trotterize(self._qb_ham, factor=1.0j*self._dt, trotter_number=self._trotter_number)

            for i, occupation in enumerate(self._ref):
                if(occupation):
                    self._nbody_counts.append(0)
        else:
            self.fill_pool()

        if(op_select_type=='residual'):
            self._eiH, self._eiH_phase = trotterize(self._qb_ham, factor=1.0j*self._dt, trotter_number=self._trotter_number)

        if self._verbose and (self._op_select_type != 'total_res'):
            print('\n\n-------------------------------------')
            print('   Second Quantized Operator Pool')
            print('-------------------------------------')
            print(self._pool_obj.str())

        self.build_orb_energies()

        if self._op_select_type == 'residual':
            self._tres_pool = [0.0 for i in range(len(self._pool))]

        avqe_iter = 0
        hit_maxiter = 0

        f = open("rsucc_.dat", "w+", buffering=1)
        f.write(f"#{'Iter(k)':>8}{'E(k)':>14}{'N(params)':>17}{'N(CNOT)':>18}{'N(measure)':>20}\n")
        f.write('#-------------------------------------------------------------------------------\n')

        while not self._converged:

            print('\n\n -----> sUCC-VP iteration ', avqe_iter, ' <-----\n')
            self.update_ansatz()

            if self._converged:
                break

            if(self._verbose):
                print('\ntoperators included from pool: \n', self._tops)
                print('\ntamplitudes for tops: \n', self._tamps)

            self.diis_solve()

            if(self._verbose):
                print('\ntamplitudes for tops post solve: \n', np.real(self._tamps))

            f.write(f'  {avqe_iter:7}    {self._energies[-1]:+15.9f}    {len(self._tamps):8}        {self._n_cnot_lst[-1]:10}        {sum(self._n_pauli_trm_measures_lst):12}\n')
            avqe_iter += 1

            if avqe_iter > self._succ_maxiter-1:
                hit_maxiter = 1
                break

        f.close()

        if hit_maxiter:
            self._Egs = self.get_final_energy(hit_max_avqe_iter=1)
            self._final_result = self._results[-1]

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

        # Print summary banner (should done for all algorithms).
        self.print_summary_banner()

        # verify that required attributes were defined
        # (should be called for all algorithms!)
        self.verify_run()

    # Define Algorithm abstract methods.
    def run_realistic(self):
        raise NotImplementedError('run_realistic() is not fully implemented for ADAPT-VQE.')

    def verify_run(self):
        self.verify_required_attributes()
        self.verify_required_VQE_attributes()
        self.verify_required_UCCVQE_attributes()

    def print_options_banner(self):
        print('\n-----------------------------------------------------')
        print('       Selected UCC Variational Projection   ')
        print('-----------------------------------------------------')

        print('\n\n               ==> sUCC-VP options <==')
        print('---------------------------------------------------------')
        # General algorithm options.
        print('Trial reference state:                   ',  ref_string(self._ref, self._nqb))
        print('Number of Hamiltonian Pauli terms:       ',  self._Nl)
        print('Trial state preparation method:          ',  self._trial_state_type)
        print('Trotter order (rho):                     ',  self._trotter_order)
        print('Trotter number (m):                      ',  self._trotter_number)
        print('Use fast version of algorithm:           ',  str(self._fast))
        if(self._fast):
            print('Measurement varience thresh:             ',  'NA')
        else:
            print('Measurement varience thresh:             ',  0.01)


        # VQE options.
        opt_thrsh_str = '{:.2e}'.format(self._res_vec_thresh)
        rsucc_thrsh_str = '{:.2e}'.format(self._rsucc_thresh)
        print('DIIS maxiter:                            ',  self._max_residual_iter)
        print('DIIS residual-norm threshold (theta):    ',  opt_thrsh_str)

        # UCCVQE options.
        print('Operator pool type:                      ',  str(self._pool_type))

        # Specific ADAPT-VQE options.
        print('sUCC-VP operator selection type:         ',  self._op_select_type)
        print('sUCC-VP residual-norm threshold (eps):   ',  rsucc_thrsh_str)
        print('sUCC-VP maxiter:                         ',  self._succ_maxiter)


    def print_summary_banner(self):

        print('\n\n                ==> sUCC-VP summary <==')
        print('-----------------------------------------------------------')
        print('Final ADAPT-VQE Energy:                     ', round(self._Egs, 10))
        print('Number of operators in pool:                 ', len(self._pool))
        print('Final number of amplitudes in ansatz:        ', len(self._tamps))
        # print('Total number of Hamiltonian measurements:    ', self.get_num_ham_measurements())
        # print('Total number of comutator measurements:      ', self.get_num_comut_measurements())
        print('Number of classical parameters used:         ', self._n_classical_params)
        print('Number of CNOT gates in deepest circuit:     ', self._n_cnot)
        print('Number of Pauli term measurements:           ', self._n_pauli_trm_measures)
        # print('Number of redidual vec evaluations:          ', self._num_res_evals)
        print('Number of residual vector evaluations:       ', self._res_vec_evals)
        print('Number of individual residual evaluations:   ', self._res_m_evals)

    # Define VQE abstract methods.
    def solve(self):
        pass

    def diis_solve(self):
        # draws heavy insiration from Daniel Smith's ccsd_diss.py code in psi4 numpy
        diis_dim = 0
        t_diis = [copy.deepcopy(self._tamps)]
        e_diis = []
        rk_norm = 1.0
        Ek0 = self.energy_feval(self._tamps)

        print('\n    k iteration         Energy               dE           Nrvec ev      Nrm ev*          ||r||')
        print('---------------------------------------------------------------------------------------------------')

        for k in range(1, self._max_residual_iter+1):
            t_old = copy.deepcopy(self._tamps)

            #do regular update
            r_k = self.get_residual_vector(self._tamps)
            rk_norm = np.linalg.norm(r_k)
            r_k = self.get_res_over_mpdenom(r_k)

            self._tamps = list(np.add(self._tamps, r_k))

            Ek = self.energy_feval(self._tamps)
            dE = Ek - Ek0
            Ek0 = Ek
            self._num_res_evals += 1
            self._res_vec_evals += 1
            self._res_m_evals += len(self._tamps)

            print(f'     {k:7}        {Ek:+12.10f}      {dE:+12.10f}      {self._res_vec_evals:4}        {self._res_m_evals:6}       {rk_norm:+12.10f}')

            if(rk_norm < self._res_vec_thresh):
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

        self._results.append('Fake result string')
        self._final_result = 'nothing'
        self._Egs = Ek

        self._energies.append(Ek)
        self._n_pauli_trm_measures_lst.append(self._n_pauli_measures_k)
        self._n_cnot_lst.append(self.build_Uvqc().get_num_cnots())

    def get_residual_vector(self, trial_amps):
        if(self._pool_type == 'sa_SD'):
            raise ValueError('Must use single term particle-hole nbody operators for residual calcultion')

        temp_pool = qforte.SQOpPool()
        for param, top in zip(trial_amps, self._tops):
            temp_pool.add_term(param, self._pool[top][1])

        A = temp_pool.get_quantum_operator('comuting_grp_lex')
        U, U_phase = trotterize(A, trotter_number=self._trotter_number)
        if U_phase != 1.0 + 0.0j:
            raise ValueError("Encountered phase change, phase not equal to (1.0 + 0.0i)")

        qc_res = qforte.QuantumComputer(self._nqb)
        qc_res.apply_circuit(self._Uprep)
        qc_res.apply_circuit(U)
        qc_res.apply_operator(self._qb_ham)
        qc_res.apply_circuit(U.adjoint())

        coeffs = qc_res.get_coeff_vec()
        residuals = []

        # each operator needs a score, so loop over toperators
        for m in self._tops:
            sq_op = self._pool[m][1]
            # occ => i,j,k,...
            # vir => a,b,c,...
            # sq_op is 1.0(a^ b^ i j) - 1.0(j^ i^ b a)

            temp_idx = sq_op.terms()[0][1][-1]
            if temp_idx < int(sum(self._ref)/2): # if temp_idx is an occupid idx
                sq_sub_tamp ,sq_sub_top = sq_op.terms()[0]
            else:
                sq_sub_tamp ,sq_sub_top = sq_op.terms()[1]

            nbody = int(len(sq_sub_top) / 2)
            destroyed = False
            denom = 1.0

            basis_I = qforte.QuantumBasis(self._nqb)
            for k, occ in enumerate(self._ref):
                basis_I.set_bit(k, occ)

            # loop over anihilators
            for p in reversed(range(nbody, 2*nbody)):
                if( basis_I.get_bit(sq_sub_top[p]) == 0):
                    destroyed=True
                    break

                basis_I.set_bit(sq_sub_top[p], 0)

            # then over creators
            for p in reversed(range(0, nbody)):
                if (basis_I.get_bit(sq_sub_top[p]) == 1):
                    destroyed=True
                    break

                basis_I.set_bit(sq_sub_top[p], 1)

            if not destroyed:

                I = basis_I.add()

                ## check for correct dets
                det_I = integer_to_ref(I, self._nqb)
                nel_I = sum(det_I)
                cor_spin_I = correct_spin(det_I, 0)

                qc_temp = qforte.QuantumComputer(self._nqb)
                qc_temp.apply_circuit(self._Uprep)
                qc_temp.apply_operator(sq_op.jw_transform())
                sign_adjust = qc_temp.get_coeff_vec()[I]

                res_m = coeffs[I] * sign_adjust # * sq_sub_tamp
                if(np.imag(res_m) > 0.0):
                    raise ValueError("residual has imaginary component, someting went wrong!!")

                residuals.append(res_m)

            else:
                raise ValueError("no ops should destroy reference, something went wrong!!")

        return residuals

    def get_res_over_mpdenom(self, residuals):

        resids_over_denoms = []

        # each operator needs a score, so loop over toperators
        for mu, m in enumerate(self._tops):
            sq_op = self._pool[m][1]

            temp_idx = sq_op.terms()[0][1][-1]
            if temp_idx < int(sum(self._ref)/2): # if temp_idx is an occupid idx
                sq_sub_tamp ,sq_sub_top = sq_op.terms()[0]
            else:
                sq_sub_tamp ,sq_sub_top = sq_op.terms()[1]

            nbody = int(len(sq_sub_top) / 2)
            destroyed = False
            denom = 0.0

            for p, op_idx in enumerate(sq_sub_top):
                if(p<nbody):
                    denom -= self._orb_e[op_idx]
                else:
                    denom += self._orb_e[op_idx]

            res_mu = copy.deepcopy(residuals[mu])
            res_mu /= denom # divide by energy denominator

            resids_over_denoms.append(res_mu)

        return resids_over_denoms

    def build_orb_energies(self):
        self._orb_e = []

        print('\nBuilding single particle energies list:')
        print('---------------------------------------')
        qc = qforte.QuantumComputer(self._nqb)
        qc.apply_circuit(build_Uprep(self._ref, 'reference'))
        E0 = qc.direct_op_exp_val(self._qb_ham)

        for i in range(self._nqb):
            qc = qforte.QuantumComputer(self._nqb)
            qc.apply_circuit(build_Uprep(self._ref, 'reference'))
            qc.apply_gate(qforte.make_gate('X', i, i))
            Ei = qc.direct_op_exp_val(self._qb_ham)

            if(i<sum(self._ref)):
                ei = E0 - Ei
            else:
                ei = Ei - E0

            print(f'  {i:3}     {ei:+16.12f}')
            self._orb_e.append(ei)

    # Define ADAPT-VQE methods.
    def update_ansatz(self):
        self._n_pauli_measures_k = 0
        if (self._op_select_type=='gradient'):
            curr_norm = 0.0
            lgrst_grad = 0.0
            Uvqc = self.build_Uvqc()

            if self._verbose:
                print('     op index (m)     N pauli terms            Gradient ')
                print('  --------------------------------------------------------')

            if self._use_comutator_grad_selection:
                grads = self.measure_comutator_gradient(self._comutator_pool, Uvqc)
            else:
                grads = self.measure_gradient(use_entire_pool=True)

            for m, grad_m in enumerate(grads):
                if self._use_comutator_grad_selection:
                    self._n_pauli_measures_k += len(self._comutator_pool.terms()[m][1].terms())
                else:
                    # referes to number of times sigma_y must be measured in "stratagies for UCC" grad eval circuit
                    self._n_pauli_measures_k += self._Nl * self._Nm[m]

                curr_norm += grad_m*grad_m
                if (self._verbose):
                    print(f'       {m:3}                {self._Nm[m]:8}             {grad_m:+12.9f}')
                if (abs(grad_m) > abs(lgrst_grad)):
                    lgrst_grad = grad_m
                    lgrst_grad_idx = m

            curr_norm = np.sqrt(curr_norm)
            if self._use_comutator_grad_selection:
                print("==> Measring gradients from pool:")
                print(" Norm of <[H,Am]> = %12.8f" %curr_norm)
                print(" Max  of <[H,Am]> = %12.8f" %lgrst_grad)
            else:
                print("==> Measring gradients:")
                print(" Norm of g_vec = %12.8f" %curr_norm)
                print(" Max  of g_vec = %12.8f" %lgrst_grad)

            self._curr_grad_norm = curr_norm
            self._grad_norms.append(curr_norm)
            self.conv_status()

            if not self._converged:

                if(self._use_cumulative_thresh):
                    temp_order_tops = []
                    grads_sq = [(grads[m] * grads[m], m) for m in range(len(grads))]
                    grads_sq.sort()
                    gm_sq_sum = 0.0
                    for m, gm_sq in enumerate(grads_sq):
                        gm_sq_sum += gm_sq[0]
                        if gm_sq_sum > (self._avqe_thresh * self._avqe_thresh):
                            print("  Adding operator m =", gm_sq[1])
                            if(gm_sq[1] not in self._tops):
                                self._tops.insert(0,gm_sq[1])
                                self._tamps.insert(0,0.0)

                    self._tops.extend(copy.deepcopy(temp_order_tops))
                    self._n_classical_params_lst.append(len(self._tops))
                else:
                    print("  Adding operator m =", lgrst_grad_idx)
                    self._tops.append(lgrst_grad_idx)
                    self._tamps.append(0.0)
                    self._n_classical_params_lst.append(len(self._tops))

            else:
                print("\n  ADAPT-VQE converged!")

        elif(self._op_select_type=='total_res'):

            self._n_pauli_measures_k += 1

            x0 = copy.deepcopy(self._tamps)
            init_gues_energy = self.energy_feval(x0)

            # do U^dag e^iH U |Phi_o> = |Phi_res>
            temp_pool = qf.SQOpPool()
            for param, top in zip(self._tamps, self._tops):
                temp_pool.add_term(param, self._pool[top][1])

            A = temp_pool.get_quantum_operator('comuting_grp_lex')
            U, U_phase = trotterize(A, trotter_number=self._trotter_number)
            if U_phase != 1.0 + 0.0j:
                raise ValueError("Encountered phase change, phase not equal to (1.0 + 0.0i)")

            qc_res = qf.QuantumComputer(self._nqb)
            qc_res.apply_circuit(self._Uprep)
            qc_res.apply_circuit(U)
            if(self._use_adaptive_t):
                self._dt = np.abs( 1.0 / (np.sqrt(2.0) * self.energy_feval(self._tamps)))
                self._eiH, self._eiH_phase = trotterize(self._qb_ham, factor=1.0j*self._dt, trotter_number=self._trotter_number)
                print(f'new dt:  {self._dt:10.8f}')

            qc_res.apply_circuit(self._eiH)
            qc_res.apply_circuit(U.adjoint())

            res_coeffs = qc_res.get_coeff_vec()
            lgrst_op_factor = 0.0

            # ned to sort the coeffs to psi_tilde
            temp_order_resids = []

            # build different res_sq list using M_omega
            if(self._M_omega != 'inf'):
                res_sq_tmp = [ np.real(np.conj(res_coeffs[I]) * res_coeffs[I]) for I in range(len(res_coeffs))]

                # Nmu_lst => [ det1, det2, det3, ... det_M_omega]
                det_lst = np.random.choice(len(res_coeffs), self._M_omega, p=res_sq_tmp)

                print(f'|Co|∆t^2 :       {np.amax(res_sq_tmp):12.14f}')
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
                self._curr_res_sq_norm = 0.0
                for rmu_sq in res_sq[:-1]:
                    self._curr_res_sq_norm += rmu_sq[0]

                self._curr_res_sq_norm /= (self._dt * self._dt)

                ## 3. print stuff
                print('  \n--> Begin selection opt with residual magnitudes:')
                print('  Initial guess energy:          ', round(init_gues_energy,10))
                print(f'  Norm of approximate res vec:  {np.sqrt(self._curr_res_sq_norm):14.12f}')

                ## 4/ check conv status (need up update function with if(M_omega != 'inf'))
                if(len(Nmu_lst)==1):
                    print('  RG-PQE converged with M_omega thresh!')
                    self._converged = 1
                    self._final_energy = self._energies[-1]
                    self._final_result = self._results[-1]
                else:
                    # print('RG-PQE did not converge!')
                    self._converged = 0

                ## 5. add new toperator
                if not self._converged:
                    if self._verbose:
                        print('\n')
                        print('     op index (Imu)     Number of times measured')
                        print('  -----------------------------------------------')
                    res_sq_sum = 0.0
                    n_ops_added = 0


                    for Nmu_tup in Nmu_lst[:-1]:
                        if(self._verbose):
                            print(f"  {Nmu_tup[1]:10}                  {np.real(Nmu_tup[0]):14}")
                        n_ops_added += 1
                        if(Nmu_tup[1] not in self._tops):
                            self._tops.insert(0,Nmu_tup[1])
                            self._tamps.insert(0,0.0)
                            self.add_op_from_basis_idx(Nmu_tup[1])

                    self._n_classical_params_lst.append(len(self._tops))

            else:
                res_sq = [( np.real(np.conj(res_coeffs[I]) * res_coeffs[I]), I) for I in range(len(res_coeffs))]

                ###
                res_sq.sort()

                self._curr_res_sq_norm = 0.0
                for rmu_sq in res_sq[:-1]:
                    self._curr_res_sq_norm += rmu_sq[0]

                self._curr_res_sq_norm /= (self._dt * self._dt)

                print('  \n--> Begin selection opt with residual magnitudes:')
                print('  Initial guess energy: ', round(init_gues_energy,10))
                print(f'  Norm of res vec:      {np.sqrt(self._curr_res_sq_norm):14.12f}')

                self.conv_status()

                if not self._converged:
                    if self._verbose:
                        print('\n')
                        print('     op index (Imu)           Residual Facotr')
                        print('  -----------------------------------------------')
                    res_sq_sum = 0.0
                    n_ops_added = 0

                    if(self._use_cumulative_thresh):
                        temp_ops = []
                        for rmu_sq in res_sq[:-1]:
                            res_sq_sum += (rmu_sq[0]/(self._dt * self._dt))
                            if res_sq_sum > (self._rsucc_thresh * self._rsucc_thresh):
                                # print("  Adding operator Imu =", rmu_sq[1])
                                if(self._verbose):
                                    print(f"  {rmu_sq[1]:10}                  {np.real(rmu_sq[0])/(self._dt * self._dt):14.12f}")
                                n_ops_added += 1
                                if(rmu_sq[1] not in self._tops):
                                    temp_ops.append(rmu_sq[1])
                                    # self._tops.insert(0,rmu_sq[1])
                                    # self._tamps.insert(0,0.0)
                                    self.add_op_from_basis_idx(rmu_sq[1])

                        for temp_op in temp_ops[::-1]:
                            self._tops.insert(0, temp_op)
                            self._tamps.insert(0, 0.0)
                            # self.add_op_from_basis_idx(temp_op)

                    else:
                        res_sq.reverse()
                        op_added = False
                        for rmu_sq in res_sq[1:]:
                            if(op_added):
                                break
                            # res_sq_sum += (rmu_sq[0]/(self._dt * self._dt))
                            # if res_sq_sum > (self._rsucc_thresh * self._rsucc_thresh):
                                # print("  Adding operator Imu =", rmu_sq[1])
                            # if(self._verbose):
                            print(f"  {rmu_sq[1]:10}                  {np.real(rmu_sq[0])/(self._dt * self._dt):14.12f}")
                            # n_ops_added += 1
                            if(rmu_sq[1] not in self._tops):
                                print('op added!')
                                self._tops.insert(0,rmu_sq[1])
                                self._tamps.insert(0,0.0)
                                self.add_op_from_basis_idx(rmu_sq[1])
                                op_added = True

                    self._n_classical_params_lst.append(len(self._tops))
                    ###

        elif (self._op_select_type == "residual"):

            self._n_pauli_measures_k += 1

            x0 = copy.deepcopy(self._tamps)
            init_gues_energy = self.energy_feval(x0)
            print('  \n--> Begin selection opt with residual magnitudes:')
            print('  Initial guess energy: ', round(init_gues_energy,10))

            if self._verbose:
                print('\n')
                print('     op index (m)          Residual Facotr')
                print('  -----------------------------------------------')

            # do U^dag e^iH U |Phi_o> = |Phi_res>

            temp_pool = qf.SQOpPool()
            for param, top in zip(self._tamps, self._tops):
                temp_pool.add_term(param, self._pool[top][1])

            A = temp_pool.get_quantum_operator('comuting_grp_lex')
            U, U_phase = trotterize(A, trotter_number=self._trotter_number)
            if U_phase != 1.0 + 0.0j:
                raise ValueError("Encountered phase change, phase not equal to (1.0 + 0.0i)")

            qc_res = qf.QuantumComputer(self._nqb)
            qc_res.apply_circuit(self._Uprep)
            qc_res.apply_circuit(U)
            qc_res.apply_circuit(self._eiH)
            qc_res.apply_circuit(U.adjoint())

            coeffs = qc_res.get_coeff_vec()
            lgrst_op_factor = 0.0

            # each operator needs a score, so loop over toperators
            for m, sq_op in enumerate(self._pool):
                sq_tamp, sq_top = sq_op
                op_factor = 0.0
                for sq_sub_tamp ,sq_sub_top in sq_top.terms():
                    basis_I = qf.QuantumBasis(self._nqb)
                    for k, occ in enumerate(self._ref):
                        basis_I.set_bit(k, occ)

                    nbody = int(len(sq_sub_top) / 2)
                    destroyed = False
                    # loop over anihilators
                    for p in reversed(range(nbody, 2*nbody)):
                        #if already un-occupied, break and don't add to op_factor
                        if( basis_I.get_bit(sq_sub_top[p]) == 0):
                            destroyed=True
                            break

                        # else set new bit to unoccupied
                        basis_I.set_bit(sq_sub_top[p], 0)

                    # then over creators
                    for p in reversed(range(0, nbody)):

                        #if already occupied, break and don't add to op_factor
                        if (basis_I.get_bit(sq_sub_top[p]) == 1):
                            destroyed=True
                            # print('destroyed!')
                            break

                        # else set new qbit to occupied
                        basis_I.set_bit(sq_sub_top[p], 1)

                    if not destroyed:
                        I = basis_I.add()
                        op_factor += np.real(sq_sub_tamp*sq_sub_tamp * np.conj(coeffs[I])*coeffs[I])


                op_factor /= (self._dt * self._dt)
                print('      ', m,  "          ",  op_factor)
                self._tres_pool[m] = op_factor

                if (abs(op_factor) > abs(lgrst_op_factor)):
                    lgrst_op_factor = op_factor
                    lgrst_op_factor_idx = m

            print("  Adding operator m =", lgrst_op_factor_idx)
            self._tops.append(lgrst_op_factor_idx)
            self._tamps.append(0.0)
            self._n_classical_params_lst.append(len(self._tops))

        else:
            raise ValueError('Invalid value specified for _op_select_type')

    def add_op_from_basis_idx(self, I):

        max_nbody = len(self._nbody_counts)

        nqb = len(self._ref)
        nel = int(sum(self._ref))

        # TODO(Nick): incorparate more flexability into this
        na_el = int(nel/2);
        nb_el = int(nel/2);

        # print('Imu: ', I)
        basis_I = qf.QuantumBasis(I)
        # print('basis Imu: ', basis_I.str(nqb))

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

        # for ( q=nel; q<nqb; q++)
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
                # for (const auto& z: parity)
                for z in parity:
                    total_parity *= z

                if(total_parity==1):
                    # particles.insert(particles.end(), holes.begin(), holes.end());
                    excitation = particles + holes
                    dexcitation = list(reversed(excitation))
                    # std::vector<> particles_adj (particles.rbegin(), particles.rend());
                    sigma_I = [1.0, tuple(excitation)]
                    # need i, j, a, b
                    # SQOperator t_temp;
                    K_temp = qf.SQOperator()
                    K_temp.add_term(+1.0, excitation);
                    K_temp.add_term(-1.0, dexcitation);
                    K_temp.simplify();
                    # this is potentially slow
                    self._pool[I] = [1.0, K_temp]
                    self._Nm.insert(0, len(K_temp.jw_transform().terms()))
                    # self._pool.insert(0, tuple(1.0, K_temp))
                    # add_term(1.0, t_temp);
                    self._nbody_counts[nbody-1] += 1

    def build_Uvqc2(self, param):
        """ This function returns the QuantumCircuit object built
        from the appropiate ampltudes (tops)

        Parameters
        ----------
        param : float
            A single parameter to opteimze appended to current _tamps.
        """
        sq_ops = []
        new_tops  = copy.deepcopy(self._tops)
        new_tamps = copy.deepcopy(self._tamps)
        new_tops.append(self._trial_op)
        new_tamps.append(param)

        temp_pool = qf.SQOpPool()
        for tamp, top in zip(new_tamps, new_tops):
            temp_pool.add_term(tamp, self._pool[top][1])

        A = temp_pool.get_quantum_operator('comuting_grp_lex')

        U, phase1 = trotterize(A, trotter_number=self._trotter_number)
        Uvqc = qf.QuantumCircuit()
        Uvqc.add_circuit(self._Uprep)
        Uvqc.add_circuit(U)
        if phase1 != 1.0 + 0.0j:
            raise ValueError("Encountered phase change, phase not equal to (1.0 + 0.0i)")

        return Uvqc

    def conv_status(self):
        # can cange to determine gradient vs residual convergence
        if abs(self._curr_res_sq_norm) < abs(self._rsucc_thresh * self._rsucc_thresh):
            self._converged = 1
            self._final_energy = self._energies[-1]
            self._final_result = self._results[-1]
        else:
            self._converged = 0

    def get_num_ham_measurements(self):
        for res in self._results:
            self._n_ham_measurements += res.nfev
        return self._n_ham_measurements

    def get_num_comut_measurements(self):
        if(self._op_select_type=='gradient'):
            self._n_comut_measurements += len(self._tamps) * len(self._pool)

        if self._use_analytic_grad:
            for m, res in enumerate(self._results):
                self._n_comut_measurements += res.njev * (m+1)

        return self._n_comut_measurements

    def get_final_energy(self, hit_max_avqe_iter=0):
        """
        Parameters
        ----------
        hit_max_avqe_iter : bool
            Wether or not to use the ADAPT-VQE has already hit the maximum
            number of iterations.
        """
        if hit_max_avqe_iter:
            print("\nADAPT-VQE at maximum number of iterations!")
            self._final_energy = self._energies[-1]
        else:
            return self._final_energy

    def get_final_result(self, hit_max_avqe_iter=0):
        """
        Parameters
        ----------
        hit_max_avqe_iter : bool
            Wether or not to use the ADAPT-VQE has already hit the maximum
            number of iterations.
        """
        if hit_max_avqe_iter:
            self._final_result = self._results[-1]
        else:
            return self._final_result
