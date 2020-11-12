"""
uccvp.py
====================================
A class for...
"""

import qforte
from qforte.abc.uccvqeabc import UCCVQE

from qforte.experiment import *
from qforte.utils.transforms import *
from qforte.utils.op_pools import *
from qforte.utils.state_prep import *
from qforte.utils.trotterization import trotterize

from qforte.helper.printing import matprint

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import lstsq

class UCCVP(UCCVQE):
    """
    A class that encompases the three componants of using the variational
    quantum eigensolver to optemize a parameterized unitary CCSD like wave function.

    UCC-VP: (1) prepares a quantum state on the quantum computer
    representing the wave function to be simulated, (2) evauates the energy by
    measurement, and (3) optemizes the wave fuction via projective solution of
    the UCC Schrodinger Equation.

    Attributes
    ----------
    _ref : list
        The set of 1s and 0s indicating the initial quantum state.

    """
    def run(self,
            opt_thresh=1.0e-5,
            opt_maxiter=200,
            optimizer='BFGS',
            pool_type='SD',
            use_analytic_grad = True,
            use_htest_gradient=False,
            use_res_solve=False,
            res_vec_thresh = 1.0e-5,
            max_residual_iter = 30,
            use_mp2_guess_amps = False,
            noise_factor = 0.0):

        # TODO (cleanup): add option to pre populate cluster amps.
        self._opt_thresh = opt_thresh
        self._opt_maxiter = opt_maxiter
        self._use_analytic_grad = use_analytic_grad
        self._optimizer = optimizer
        self._pool_type = pool_type

        self._use_res_solve = use_res_solve
        self._use_htest_gradient = use_htest_gradient
        self._res_vec_thresh = res_vec_thresh
        self._max_residual_iter = max_residual_iter

        self._use_mp2_guess_amps = use_mp2_guess_amps

        self._noise_factor = noise_factor

        self._tops = []
        self._tamps = []
        self._comutator_pool = []
        self._converged = 0

        self._res_vec_evals = 0
        self._res_m_evals = 0

        self._n_classical_params = 0
        self._n_cnot = 0
        self._n_pauli_trm_measures = 0

        self._n_pauli_measures_k = 0

        self._results = [] # remove this

        # Print options banner (should done for all algorithms).
        self.print_options_banner()

        ######### UCCSD-VQE #########

        self.fill_pool()

        if self._verbose:
            print('\n\n-------------------------------------')
            print('   Second Quantized Operator Pool')
            print('-------------------------------------')
            print(self._pool_obj.str())

        self.initialize_ansatz()

        if(self._verbose):
            print('\nt operators included from pool: \n', self._tops)
            print('Initial tamplitudes for tops: \n', self._tamps)

        if self._use_res_solve:
            self.build_orb_energies()
            self.diis_solve()
        else:
            self.solve()

        if(self._verbose):
            print('\nt operators included from pool: \n', self._tops)

            print('Final tamplitudes for tops:')
            print('------------------------------')
            for i, tamp in enumerate( self._tamps ):
                print(f'  {i:4}      {tamp:+12.8f}')

        ######### UCCSD-VQE #########
        self._n_nonzero_params = 0
        for tmu in self._tamps:
            if(np.abs(tmu) > 1.0e-12):
                self._n_nonzero_params += 1

        # Print summary banner (should done for all algorithms).
        self.print_summary_banner()

        # verify that required attributes were defined
        # (should be called for all algorithms!)
        self.verify_run()

    # Define Algorithm abstract methods.
    def run_realistic(self):
        raise NotImplementedError('run_realistic() is not fully implemented for UCCSD-VQE.')

    def verify_run(self):
        self.verify_required_attributes()
        self.verify_required_VQE_attributes()
        self.verify_required_UCCVQE_attributes()

    def print_options_banner(self):
        print('\n-----------------------------------------------------')
        print('           Unitary Coupled Cluster VP   ')
        print('-----------------------------------------------------')

        print('\n\n                 ==> UCC-VP options <==')
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
        opt_thrsh_str = '{:.2e}'.format(self._opt_thresh)
        print('Optimization algorithm:                  ',  self._optimizer)
        print('Optimization maxiter:                    ',  self._opt_maxiter)
        print('Optimizer grad-norm threshold (theta):   ',  opt_thrsh_str)

        # UCCVQE options.
        print('Use analytic gradient:                   ',  str(self._use_analytic_grad))
        print('Operator pool type:                      ',  str(self._pool_type))


    def print_summary_banner(self):

        print('\n\n                   ==> UCC-VP summary <==')
        print('-----------------------------------------------------------')
        print('Final UCCSD-VQE Energy:                     ', round(self._Egs, 10))
        print('Number of operators in pool:                 ', len(self._pool))
        print('Final number of amplitudes in ansatz:        ', len(self._tamps))
        # print('Total number of Hamiltonian measurements:    ', self.get_num_ham_measurements())
        # print('Total number of comutator measurements:      ', self.get_num_comut_measurements())
        print('Number of classical parameters used:         ', len(self._tamps))
        print('Number of non-zero parameters used:          ', self._n_nonzero_params)
        print('Number of CNOT gates in deepest circuit:     ', self._n_cnot)
        # print('Number of Pauli term measurements:           ', self._n_pauli_trm_measures)
        print('Number of residual vector evaluations:       ', self._res_vec_evals)
        print('Number of individual residual evaluations:   ', self._res_m_evals)

    # Define VQE abstract methods.
    def solve(self):
        """
        Parameters
        ----------
        fast : bool
            Wether or not to use the optemized but unphysical energy evaluation
            function.
        maxiter : int
            The maximum number of iterations for the scipy optimizer.
        """

        opts = {}
        opts['gtol'] = self._opt_thresh
        opts['disp'] = False
        if(self._verbose):
            opts['disp'] = True
        opts['maxiter'] = self._opt_maxiter
        x0 = copy.deepcopy(self._tamps)
        init_gues_energy = self.energy_feval(x0)

        if self._use_analytic_grad:
            print('  \n--> Begin opt with analytic graditent:')
            print('  Initial guess energy: ', round(init_gues_energy,1000))
            res =  minimize(self.energy_feval, x0,
                                    method=self._optimizer,
                                    jac=self.gradient_ary_feval,
                                    options=opts)

        else:
            print('  \n--> Begin opt with grad estimated using first-differences:')
            print('  Initial guess energy: ', round(init_gues_energy,1000))
            res =  minimize(self.energy_feval, x0,
                                    method=self._optimizer,
                                    options=opts)

        if(res.success):
            print('  minimization successful.')
            print('  min Energy: ', res.fun)
            self._Egs = res.fun
            self._final_result = res
            self._tamps = list(res.x)
        else:
            print('  WARNING: minimization result may not be tightly converged.')
            print('  min Energy: ', res.fun)
            self._Egs = res.fun
            self._final_result = res
            self._tamps = list(res.x)

        self._n_classical_params = self._n_classical_params = len(self._tamps)
        self._n_cnot = self.build_Uvqc().get_num_cnots()
        self._n_pauli_trm_measures += self._Nl * res.nfev
        # for m in range(self._n_classical_params):
        #     self._n_pauli_trm_measures += len(self._comutator_pool.terms()[m][1].terms()) * res.njev

    def diis_solve(self):
        # draws heavy insiration from Daniel Smith's ccsd_diss.py code in psi4 numpy
        diis_dim = 0

        t_diis = [copy.deepcopy(self._tamps)]
        e_diis = []
        # k_counter = 0
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

                # could be more efficient
                for i, ei in enumerate(e_diis):
                    for j, ej in enumerate(e_diis):
                        B[i,j] = np.dot(np.real(ei), np.real(ej))

                B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()

                x = np.linalg.solve(B, bsol)

                t_new = np.zeros(( len(self._tamps) ))
                for l in range(diis_dim):
                    temp_ary = x[l] * np.asarray(t_diis[l+1])
                    t_new = np.add(t_new, temp_ary)

                self._tamps = copy.deepcopy(t_new)

        self._results.append('Fake result string')
        self._final_result = 'nothing'
        self._Egs = Ek


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

                if(self._noise_factor > 1e-12):
                    res_m = np.random.normal(np.real(res_m), self._noise_factor)

                residuals.append(res_m)

            else:
                raise ValueError("no ops should destroy reference, something went wrong!!")

        return residuals

    def get_res_over_mpdenom(self, residuals):

        resids_over_denoms = []

        # each operator needs a score, so loop over toperators
        for m in self._tops:
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

            res_m = copy.deepcopy(residuals[m])
            res_m /= denom # divide by energy denominator

            resids_over_denoms.append(res_m)

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

    def initialize_ansatz(self):
        for l in range(len(self._pool)):
            self._tops.append(l)
            self._tamps.append(0.0)

        if self._use_mp2_guess_amps:
            self._tamps = self.get_mp2_guess_amps()

    def get_num_ham_measurements(self):
        self._n_ham_measurements = self._final_result.nfev
        return self._n_ham_measurements

    def get_num_comut_measurements(self):
        if self._use_analytic_grad:
            self._n_comut_measurements = self._final_result.njev * (len(self._pool))
            return self._n_comut_measurements
        else:
            return 0

    ### totally junk experimental functions
    def get_mp2_guess_amps(self):

        if(self._pool_type != 'SD'):
            raise ValueError('Must use particle-hole singles and doubles pool to enable mp2 cluster amplitude guess')

        guess_amps = []

        print('of ham\n\n', self._sys._sq_of_hamiltonian.terms)

        for m in self._tops:
            sq_op = self._pool[m][1]
            sq_sub_tamp, sq_sub_top = sq_op.terms()[0]

            nbody = int(len(sq_sub_top) / 2)
            destroyed = False

            if(nbody==1):
                guess_amps.append( 0.0 )

            if(nbody==2):
                terms_tup1 = ((sq_sub_top[2], 1), (sq_sub_top[3], 1), (sq_sub_top[1], 0), (sq_sub_top[0],0) )
                terms_tup2 = ((sq_sub_top[2], 1), (sq_sub_top[3], 1), (sq_sub_top[0], 0), (sq_sub_top[1],0) )
                num  = self._sys._sq_of_hamiltonian.terms[terms_tup1]
                num -= self._sys._sq_of_hamiltonian.terms[terms_tup2]

                denom = 0.0
                denom -= self._orb_e[sq_sub_top[0]] #e_a
                denom -= self._orb_e[sq_sub_top[1]] #e_b
                denom += self._orb_e[sq_sub_top[2]] #e_i
                denom += self._orb_e[sq_sub_top[3]] #e_j

                guess_amps.append(num/denom)

        return guess_amps
