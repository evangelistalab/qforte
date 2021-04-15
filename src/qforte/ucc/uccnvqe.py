"""
uccnvqe.py
====================================
A class for using the variational quantum eigensolver
with a disentangled (Trotterizerd) UCCN anxatz with fixed
excitaion order.
"""

import qforte
from qforte.abc.uccvqeabc import UCCVQE

from qforte.experiment import *
from qforte.utils.transforms import *
from qforte.utils.op_pools import *
from qforte.utils.state_prep import *
from qforte.utils.trotterization import trotterize

import numpy as np
from scipy.optimize import minimize

class UCCNVQE(UCCVQE):
    """
    A class that encompasses the three components of using the variational
    quantum eigensolver to optimize a parameterized disentangled UCCN-like
    wave function. (1) prepares a quantum state on the quantum computer
    representing the wave function to be simulated, (2) evauates the energy and
    gradients (3) optemizes the the wave funciton by minimizing the energy.

    """
    def run(self,
            opt_thresh=1.0e-5,
            opt_ftol=1.0e-5,
            opt_maxiter=200,
            pool_type='SD',
            optimizer='BFGS',
            use_analytic_grad = True,
            noise_factor = 0.0):

        """ Runs the entire algorithm. Usually called by the user in a script.

        Parameters
        ----------
        opt_thresh : float
            The numerical convergence threshold for the specified classical
            optimization algorithm. Is usually the norm of the gradient, but
            is algorithm dependant, see scipy.minimize.optimize for detials.

        opt_ftol : float
            An alterative convergence threshold for optimization algorithms
            that do not rely on gradients such as direct-search algorithms.
            Only used if such an algorithm is specified.

        opt_maxiter : int
            The maximum number of iterations for the classical optimizer.

        optimizer : string
            The string specifying what classical optimization algorithm will be used.

        use_analytic_grad : bool
            Whether or not to use an analytic function for the gradient to pass to
            the optimizer. If false, the optimizer will use self-generated approximate
            gradients (if BFGS algorithm is used).

        pool_type : string
            Specifies the kinds of tamplitudes allowed in the UCCN-VQE
            parameterization.
                SA_SD: At most two orbital excitations. Assumes a singlet wavefunction and closed-shell Slater determinant reducing the number of amplitudes.

                SD: At most two orbital excitations.

                SDT: At most three orbital excitations.

                SDTQ: At most four orbital excitations.

                SDTQP: At most five orbital excitations.

                SDTQPH: At most six orbital excitations.

        """

        self._opt_thresh = opt_thresh
        self._opt_ftol = opt_ftol
        self._opt_maxiter = opt_maxiter
        self._use_analytic_grad = use_analytic_grad
        self._optimizer = optimizer
        self._pool_type = pool_type
        self._noise_factor = noise_factor

        self._tops = []
        self._tamps = []
        self._conmutator_pool = []
        self._converged = 0

        self._n_classical_params = 0
        self._n_cnot = 0
        self._n_pauli_trm_measures = 0

        # Print options banner (should done for all algorithms).
        self.print_options_banner()

        ######### UCCN-VQE #########

        self.fill_pool()

        if self._verbose:
            self._pool_obj.print_pool()

        self.initialize_ansatz()

        if(self._verbose):
            print('\nt operators included from pool: \n', self._tops)
            print('\nInitial tamplitudes for tops: \n', self._tamps)

        self.solve()

        if(self._verbose):
            print('\nt operators included from pool: \n', self._tops)
            print('\nFinal tamplitudes for tops: \n', self._tamps)

        ######### UCCSD-VQE #########
        self._n_nonzero_params = 0
        for tmu in self._tamps:
            if(np.abs(tmu) > 1.0e-12):
                self._n_nonzero_params += 1

        # verify that required attributes were defined
        # (should be called for all algorithms!)
        self.verify_run()

    # Define Algorithm abstract methods.
    def run_realistic(self):
        raise NotImplementedError('run_realistic() is not fully implemented for UCCN-VQE.')

    def verify_run(self):
        self.verify_required_attributes()
        self.verify_required_VQE_attributes()
        self.verify_required_UCCVQE_attributes()

    def print_options_banner(self):
        print('\n-----------------------------------------------------')
        print('          Unitary Coupled Cluster VQE   ')
        print('-----------------------------------------------------')

        print('\n\n               ==> UCCN-VQE options <==')
        print('---------------------------------------------------------')
        # General algorithm options.
        print('Trial reference state:                   ',  ref_string(self._ref, self._nqb))
        print('Number of Hamiltonian Pauli terms:       ',  self._Nl)
        print('Trial state preparation method:          ',  self._trial_state_type)
        print('Trotter order (rho):                     ',  self._trotter_order)
        print('Trotter number (m):                      ',  self._trotter_number)
        print('Use fast version of algorithm:           ',  str(self._fast))
        if(self._fast):
            print('Measurement variance thresh:             ',  'NA')
        else:
            print('Measurement variance thresh:             ',  0.01)


        # VQE options.
        opt_thrsh_str = '{:.2e}'.format(self._opt_thresh)
        print('Optimization algorithm:                  ',  self._optimizer)
        print('Optimization maxiter:                    ',  self._opt_maxiter)
        print('Optimizer grad-norm threshold (theta):   ',  opt_thrsh_str)

        # UCCVQE options.
        print('Use analytic gradient:                   ',  str(self._use_analytic_grad))
        print('Operator pool type:                      ',  str(self._pool_type))


    def print_summary_banner(self):

        print('\n\n                ==> UCCN-VQE summary <==')
        print('-----------------------------------------------------------')
        print('Final UCCN-VQE Energy:                      ', round(self._Egs, 10))
        print('Number of operators in pool:                 ', len(self._pool))
        print('Final number of amplitudes in ansatz:        ', len(self._tamps))
        print('Total number of Hamiltonian measurements:    ', self.get_num_ham_measurements())
        print('Total number of commutator measurements:     ', self.get_num_commut_measurements())
        print('Number of classical parameters used:         ', self._n_classical_params)
        print('Number of non-zero parameters used:          ', self._n_nonzero_params)
        print('Number of CNOT gates in deepest circuit:     ', self._n_cnot)
        print('Number of Pauli term measurements:           ', self._n_pauli_trm_measures)

        print('Number of grad vector evaluations:           ', self._grad_vec_evals)
        print('Number of individual grad evaluations:       ', self._grad_m_evals)

    # Define VQE abstract methods.
    def solve(self):
        # Construct arguments to hand to the minimizer.
        opts = {}
        opts['gtol'] = self._opt_thresh

        opts['fatol'] = self._opt_ftol
        opts['ftol'] = self._opt_ftol
        opts['tol'] = self._opt_ftol

        opts['disp'] = True
        opts['maxiter'] = self._opt_maxiter
        opts['maxfun']  = self._opt_maxiter

        x0 = copy.deepcopy(self._tamps)
        init_gues_energy = self.energy_feval(x0)

        if self._use_analytic_grad:
            print('  \n--> Begin opt with analytic gradient:')
            print(f" Initial guess energy:              {init_gues_energy:+12.10f}")
            res =  minimize(self.energy_feval, x0,
                                    method=self._optimizer,
                                    jac=self.gradient_ary_feval,
                                    options=opts,
                                    callback=self.report_iteration)

            # account for paulit term measurement for gradient evaluations
            for m in range(len(self._tamps)):
                self._n_pauli_trm_measures += self._Nm[m] * self._Nl * res.njev

        else:
            print('  \n--> Begin opt with grad estimated using first-differences:')
            print(f" Initial guess energy:              {init_gues_energy:+12.10f}")
            res =  minimize(self.energy_feval, x0,
                                    method=self._optimizer,
                                    options=opts,
                                    callback=self.report_iteration)

        if(res.success):
            print('  => Minimization successful!')
            print(f'  => Minimum Energy: {res.fun:+12.10f}')
            self._Egs = res.fun
            if(self._optimizer == 'POWELL'):
                print(type(res.fun))
                print(res.fun)
                self._Egs = res.fun[()]
            self._final_result = res
            self._tamps = list(res.x)
        else:
            print('  => WARNING: minimization result may not be tightly converged.')
            print(f'  => Minimum Energy: {res.fun:+12.10f}')
            self._Egs = res.fun
            if(self._optimizer == 'POWELL'):
                print(type(res.fun))
                print(res.fun)
                self._Egs = res.fun[()]
            self._final_result = res
            self._tamps = list(res.x)

        self._n_classical_params = len(self._tamps)
        self._n_cnot = self.build_Uvqc().get_num_cnots()

        # account for pauli term measurement for energy evaluations
        self._n_pauli_trm_measures += self._Nl * res.nfev


    def initialize_ansatz(self):
        """ Adds all operators in the pool to the list of operators in the
        circuit, with amplitude 0.

        """
        self._tops = list(range(len(self._pool)))
        self._tamps = [0.0] * len(self._pool)

    def get_num_ham_measurements(self):
        self._n_ham_measurements = self._final_result.nfev
        return self._n_ham_measurements

    def get_num_commut_measurements(self):
        # TODO: depricate this funciton
        # if self._use_analytic_grad:
        #     self._n_commut_measurements = self._final_result.njev * (len(self._pool))
        #     return self._n_commut_measurements
        # else:
        #     return 0
        return 0
