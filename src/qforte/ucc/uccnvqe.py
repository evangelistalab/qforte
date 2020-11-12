"""
uccsdvqe.py
====================================
A class for using an experiment to execute the variational quantum eigensolver
for a trotterized UCCSD anxatz.
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
    A class that encompases the three componants of using the variational
    quantum eigensolver to optemize a parameterized unitary CCSD like wave function.

    UCCSD-VQE: (1) prepares a quantum state on the quantum computer
    representing the wave function to be simulated, (2) evauates the energy by
    measurement, and (3) optemizes the the wave funciton by minimizing the energy.

    Attributes
    ----------
    _ref : list
        The set of 1s and 0s indicating the initial quantum state.

    _nqubits : int
        The number of qubits the calculation empolys.

    _operator : QuantumOperator
        The operator to be measured (usually the Hamiltonain), mapped to a
        qubit representation.

    _avqe_thresh : float
        The gradient norm threshold to determine when the UCCSD-VQE
        algorithm has converged.

    _opt_thresh : float
        The gradient norm threshold to determine when the classical optemizer
        algorithm has converged.

    _use_fast_measurement : bool
        Whether or not to use a faster version of the algorithm that bypasses
        measurment (unphysical for quantum computer).

    _use_analytic_grad : bool
        Whether or not to use an alaystic function for the gradient to pass to
        the optemizer. If false, the optemizer will use self-generated approximate
        gradients (if BFGS algorithm is used).

    _optimizer : string
        The type of opterizer to use for the classical portion of VQE. Suggested
        algorithms are 'BFGS' or 'Nelder-Mead' although there are many options
        (see SciPy.optemize.minimize documentation).

    _trott_num : int
        The Trotter number for the calculation
        (exact in the infinte limit).

    _results : list
        The optemizer result objects from each iteration of UCCSD-VQE.

    _energies : list
        The optemized energies from each iteration of UCCSD-VQE.

    _grad_norms : list
        The gradient norms from each iteration of UCCSD-VQE.

    _curr_grad_norm : float
        The gradient norm for the current iteration of UCCSD-VQE.

    _initial_guess_energy
        The initial guess energy from each iteration of UCCSD-VQE.

    _pool_obj : SDOpPool
        An SDOpPool object corresponding to the specefied operators of
        interest.

    _pool : list of lists with tuple and float
        The list of (optionally symmetrized) singe and double excitation
        operators to consizer. represented in the form,
        [ [(p,q), t_pq], .... , [(p,q,s,r), t_pqrs], ... ]
        where p, q, r, s are idicies of normal ordered creation or anihilation
        operators.

    _tops : list
        A list of indicies representing selected operators in the pool.

    _tamps : list
        A list of amplitudes (to be optemized) representing selected
        operators in the pool.

    _comutator_pool : list
        The QuantumOperator objects representing the comutators [H, Am] of the
        Hamiltonian (H) and each member of the operator pool (Am).

    _N_samples : int
        The number of times to measure each term in _operator
        (not yet functional).

    _converged : bool
        Whether or not the UCCSD-VQE has converged according to the gradient-norm
        threshold.

    _final_energy : float
        The final UCCSD-VQE energy value.

    _final_result : Result
        The last result object from the optemizer.

    _n_ham_measurements : int
        The total number of times the energy was evaluated via
        measurement of the Hamiltoanin

    _n_comut_measurements : int
        The total number of times the comutator was evaluated via
        measurement of [H, Am].


    Methods
    -------
    fill_pool()
        Fills the pool_ with indicies pertaining spin-complete, single and
        double excitation operators according to _nocc and _nvir.

    fill_comutator_pool()
        Fills the _comutator_pool with circuits considering the _operator to
        be measured and the _pool.

    initialize_ansatz()
        Adds a paramater and operator to the UCCSD-VQE circuit, and checks for
        convergence.

    build_Uprep()
        Returns a QuantumCircuit object corresponding to the state preparation
        circuit for the UCCSD-VQE ansatz on a given iteration.

    measure_gradient()
        Returns the measured energy gradient with respect to a single
        paramater Am.

    measure_energy()
        Returns the measured energy.

    energy_feval()
        Builds a state preparation circuit given a parameter list and returns
        the measured energy. Used as the function the minimizer calls.

    gradient_ary_feval()
        Computes the gradients with respect to all operators currently in the
        UCCSD-VQE ansatz. Used as the jacobian the minimizer calls.

    solve()
        Runs the optimizer to mimimize the energy. Sets certain optimizer
        parameters internally.

    conv_status()
        Sets the convergence states.

    get_num_ham_measurements()
        Returns the total number of times the energy was evaluated via
        measurement of the Hamiltoanin.

    get_num_comut_measurements()
        Returns the total number of times the comutator was evaluated via
        measurement of [H, Am].

    get_final_energy()
        Returns the final energy.

    get_final_result()
        Retruns the fianl optemization result from the optemizer. Contains
        the final amplitudes used.
    """
    def run(self,
            opt_thresh=1.0e-5,
            opt_ftol=1.0e-5,
            opt_maxiter=200,
            pool_type='SD',
            optimizer='BFGS',
            use_analytic_grad = True,
            noise_factor = 0.0):

        self._opt_thresh = opt_thresh
        self._opt_ftol = opt_ftol
        self._opt_maxiter = opt_maxiter
        self._use_analytic_grad = use_analytic_grad
        self._optimizer = optimizer
        self._pool_type = pool_type
        self._noise_factor = noise_factor

        self._tops = []
        self._tamps = []
        self._comutator_pool = []
        self._converged = 0

        self._n_classical_params = 0
        self._n_cnot = 0
        self._n_pauli_trm_measures = 0

        # Print options banner (should done for all algorithms).
        self.print_options_banner()

        ######### UCCSD-VQE #########

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
        print('          Unitary Coupled Cluster VQE   ')
        print('-----------------------------------------------------')

        print('\n\n                 ==> UCC-VQE options <==')
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

        print('\n\n                  ==> UCC-VQE summary <==')
        print('-----------------------------------------------------------')
        print('Final UCCSD-VQE Energy:                     ', round(self._Egs, 10))
        print('Number of operators in pool:                 ', len(self._pool))
        print('Final number of amplitudes in ansatz:        ', len(self._tamps))
        print('Total number of Hamiltonian measurements:    ', self.get_num_ham_measurements())
        print('Total number of comutator measurements:      ', self.get_num_comut_measurements())
        print('Number of classical parameters used:         ', self._n_classical_params)
        print('Number of non-zero parameters used:          ', self._n_nonzero_params)
        print('Number of CNOT gates in deepest circuit:     ', self._n_cnot)
        print('Number of Pauli term measurements:           ', self._n_pauli_trm_measures)

        print('Number of grad vector evaluations:           ', self._grad_vec_evals)
        print('Number of individual grad evaluations:       ', self._grad_m_evals)

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

        opts['fatol'] = self._opt_ftol
        opts['ftol'] = self._opt_ftol
        opts['tol'] = self._opt_ftol

        opts['disp'] = True
        opts['maxiter'] = self._opt_maxiter
        opts['maxfun']  = self._opt_maxiter

        x0 = copy.deepcopy(self._tamps)
        init_gues_energy = self.energy_feval(x0)

        if self._use_analytic_grad:
            print('  \n--> Begin opt with analytic graditent:')
            print(f" Initail guess energy:              {init_gues_energy:+12.10f}")
            res =  minimize(self.energy_feval, x0,
                                    method=self._optimizer,
                                    jac=self.gradient_ary_feval,
                                    options=opts,
                                    callback=self.callback)

            # account for paulit term measurement for gradient evaluations
            for m in range(len(self._tamps)):
                self._n_pauli_trm_measures += self._Nm[m] * self._Nl * res.njev

        else:
            print('  \n--> Begin opt with grad estimated using first-differences:')
            print(f" Initail guess energy:              {init_gues_energy:+12.10f}")
            res =  minimize(self.energy_feval, x0,
                                    method=self._optimizer,
                                    options=opts,
                                    callback=self.callback)

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
        for l in range(len(self._pool)):
            self._tops.append(l)
            self._tamps.append(0.0)

    def get_num_ham_measurements(self):
        self._n_ham_measurements = self._final_result.nfev
        return self._n_ham_measurements

    def get_num_comut_measurements(self):
        # if self._use_analytic_grad:
        #     self._n_comut_measurements = self._final_result.njev * (len(self._pool))
        #     return self._n_comut_measurements
        # else:
        #     return 0
        return 0
