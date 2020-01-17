"""
adaptvqe.py
====================================
A class for using an experiment to execute the variational quantum eigensolver
for the Adaptive Derivative-Assembled Pseudo-Trotter anxatz.
"""

import qforte
from qforte.experiment import *
from qforte.utils.transforms import *
from qforte.utils.op_pools import *
from qforte.ucc.ucc_helpers import *
import numpy as np
import scipy
from scipy.optimize import minimize

class ADAPTVQE(object):
    """
    A class that encompases the three componants of using the variational
    quantum eigensolver to optemize a parameterized unitary CC like wave function.
    Over a number of iterations the VQE: (1) prepares a quantum state on the quantum computer
    representing the wave function to be simulated, (2) evauates the energy by
    measurement, and (3) optemizes the the wave funciton by minimizing the energy.
    VQE object constructor.

    Attributes
    ----------
    _ref : list
        The set of 1s and 0s indicating the initial quantum state.

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

    _operator : QuantumOperator
        The operator to be measured (usually the Hamiltonain), mapped to a
        qubit representation.

    _N_samples : int
        The number of times to measure each term in _operator.
    _alredy_anti_herm : bool
        Wether or not Top is passed already in an anti-hermetian form.
    _use_symmetric_amps : bool
        Reduce the number of paramaters to be optemized by enforcing the
        symmetry of certain amplitudes.
    _many_preps : bool
        Use a new state preparation for every measurement.
        (Not yet functional).
    _optimizer : str
        The scipy optimizer to be used for the VQE.

    Methods
    -------
    XX unpack_args()
        Takes the second quantized operator and splits it into a list of
        amplitueds and a list of tuples containing normal ordered anihilator
        and creator indicies.

    """

    #TODO: Fix N_samples arg in Experiment class to only be take for finite measurement
    def __init__(self, ref, operator, avqe_thresh,
                 opt_thresh = 1.0e-5,
                 N_samples = 100,
                 alredy_anti_herm = False,
                 use_symmetric_amps = False,
                 use_fast_measurement= True,
                 many_preps = False,
                 optimizer= 'BFGS',
                 use_analytic_grad = True,
                 trott_num = 1):
        """
        Parameters
        ----------
        ref : list
            The set of 1s and 0s indicating the initial quantum state
        Top : list of lists with tuple and float
            The cluster opterator T in a second quantized format, it is
            represented in the form
            [ [(p,q), t_pq], .... , [(p,q,s,r), t_pqrs], ... ]
            where p, q, r, s are idicies of normal ordered creation or anihilation
            operators.
        operator : QuantumOperator
            The operator to be measured (usually the Hamiltonain), mapped to a
            qubit representation.
        N_samples : int
            The number of times to measure each term in operator.
        alredy_anti_herm : bool
            Wether or not Top is passed already in an anti-hermetian form.
        use_symmetric_amps : bool
            Reduce the number of paramaters to be optemized by enforcing the
            symmetry of certain amplitudes. (Not yet functional).
        use_analytic_grad : bool
            Use the get_gradients funciton to reduce the number of function
            calls in the optimizer.
        many_preps : bool
            Use a new state preparation for every measurement.
            (Not yet functional).
        optimizer : str
            The scipy optimizer to be used for the VQE.
        """
        #TODO(Nick): Elimenate getting info about nqubits in the 'len(ref)' fashion
        self._ref = ref
        self._nqubtis = len(ref)
        self._operator = operator
        self._avqe_thresh = avqe_thresh
        self._opt_thresh = opt_thresh
        self._N_samples = N_samples
        self._use_symmetric_amps = use_symmetric_amps
        self._use_fast_measurement = use_fast_measurement
        self._use_analytic_grad = use_analytic_grad
        self._many_preps = many_preps
        self._optimizer = optimizer
        self._trott_num = trott_num

        self._results = []
        self._energies = []
        self._grad_norms = []
        self._initial_guess_energy = []
        self._tops = []
        self._tamps = []
        self._comutator_pool = []
        self._converged = 0


    def fill_pool(self, nocc=None, nvir=None):
        #TODO(correct path ot SDOpPool fliles)
        self._pool_obj = SDOpPool(self._ref, nocc=nocc, nvir=nvir, multiplicity = 0, order = 2)
        self._pool_obj.fill_pool()
        self._pool = self._pool_obj.get_pool_lst()

    def fill_comutator_pool(self):
        print('--> Building comutators for gradient measurement:')
        for i in range(len(self._pool)):
            Am_org = get_ucc_jw_organizer(self._pool[i], already_anti_herm=True)
            H_org = circuit_to_organizer(self._operator)
            HAm_org = join_organizers(H_org, Am_org)
            HAm = organizer_to_circuit(HAm_org) # actually returns a single-term QuantumOperator
            self._comutator_pool.append(HAm)
        print('  comutators complete.')

    def update_ansatz(self): ###
        curr_norm = 0.0
        lgrst_grad = 0.0
        Uprep = self.build_Uprep()
        for m, HAm in enumerate(self._comutator_pool):
            """Here HAm is in QuantumOperator form"""
            grad_m = self.measure_gradient(HAm, Uprep)
            curr_norm += grad_m*grad_m
            if abs(grad_m) > abs(lgrst_grad):
                lgrst_grad = grad_m
                lgrst_grad_idx = m

        curr_norm = np.sqrt(curr_norm)
        print("--> Measring gradients from pool:")
        print(" Norm of <[H,Am]> = %12.8f" %curr_norm)
        print(" Max  of <[H,Am]> = %12.8f" %lgrst_grad)

        self._curr_grad_norm = curr_norm
        self._grad_norms.append(curr_norm)
        self.conv_status()

        if not self._converged:
            print(" Adding operator m =", lgrst_grad_idx)
            self._tops.append(lgrst_grad_idx)
            self._tamps.append(0.0)
        else:
            print("\n ADAPT-VQE converged!")

    def build_Uprep(self, params=None):
        """ This function returns the QuantumCircuit object built
        from the appropiate ampltudes (tops)

        """
        sq_ops = []
        for j, pool_idx in enumerate(self._tops):
            appended_term = copy.deepcopy(self._pool[pool_idx])
            for l in range(len(appended_term)):
                if params is None:
                    appended_term[l][1] *= self._tamps[j]
                else:
                    appended_term[l][1] *= params[j]
                sq_ops.append(appended_term[l])

        Uorg = get_ucc_jw_organizer(sq_ops, already_anti_herm=True)
        A = organizer_to_circuit(Uorg)
        temp_op1 = qforte.QuantumOperator() # A temporary operator to multiply H by
        for t in A.terms():
            c, op = t
            phase =  -c
            temp_op1.add_term(phase, op)

        U, phase1 = qforte.trotterization.trotterize(temp_op1, trotter_number=self._trott_num)
        Uprep = qforte.QuantumCircuit()
        for j in range(len(self._ref)):
            if self._ref[j] == 1:
                Uprep.add_gate(qforte.make_gate('X', j, j))

        Uprep.add_circuit(U)
        if phase1 != 1.0 + 0.0j:
            raise ValueError("Encountered phase change, phase not equal to (1.0 + 0.0i)")

        return Uprep

    def measure_gradient(self, HAm, Ucirc):
        # TODO: Write so finite measurement can be used
        if self._use_fast_measurement:
            myQC = qforte.QuantumComputer(self._nqubtis)
            myQC.apply_circuit(Ucirc)
            val = 2*np.real(myQC.direct_op_exp_val(HAm))
        else:
            Exp = qforte.Experiment(self._nqubtis, Ucirc, HAm, self._N_samples)
            empty_params = []
            val = 2*Exp.perfect_experimental_avg(empty_params)

        assert(np.isclose(np.imag(val),0.0))
        return val

    def measure_energy(self, Ucirc):
        if self._use_fast_measurement:
            myQC = qforte.QuantumComputer(self._nqubtis)
            myQC.apply_circuit(Ucirc)
            val = np.real(myQC.direct_op_exp_val(self._operator))
        else:
            Exp = qforte.Experiment(self._nqubtis, Ucirc, self._operator, self._N_samples)
            empty_params = []
            val = Exp.perfect_experimental_avg(empty_params)

        assert(np.isclose(np.imag(val),0.0))
        return val

    def energy_feval(self, params):
        Ucirc = self.build_Uprep(params=params)
        return self.measure_energy(Ucirc)

    def gradient_ary_feval(self, params):
        Uprep = self.build_Uprep(params=params)
        grad_lst = []
        for m in self._tops:
            grad_lst.append(self.measure_gradient(self._comutator_pool[m], Uprep))

        return np.asarray(grad_lst)


    def solve(self, fast=False, opt_maxiter=200):
        """Runs the optimizer to mimimize the energy. Sets certain optimizer parameters
        internally.

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
        opts['maxiter'] = opt_maxiter
        x0 = copy.deepcopy(self._tamps)
        init_gues_energy = self.energy_feval(x0)
        self._initial_guess_energy.append(init_gues_energy)

        if self._use_analytic_grad:
            print('  \n--> Begin opt with analytic graditent:')
            print('  Initial guess energy: ', round(init_gues_energy,10))
            res =  minimize(self.energy_feval, x0,
                                    method=self._optimizer,
                                    jac=self.gradient_ary_feval,
                                    options=opts)

        else:
            print('  \n--> Begin opt with grad estimated using first-differences:')
            print('  Initial guess energy: ', round(init_gues_energy,10))
            res =  minimize(self.energy_feval, x0,
                                    method=self._optimizer,
                                    options=opts)

        if(res.success):
            print('  minimization successful.')
            print('  min Energy: ', res.fun)
            self._energies.append(res.fun)
            self._results.append(res)
            self._tamps = list(res.x)
        else:
            print('  WARNING: minimization result may not be tightly converged.')
            print('  min Energy: ', res.fun)
            self._energies.append(res.fun)
            self._results.append(res)
            self._tamps = list(res.x)

    def conv_status(self):
        if abs(self._curr_grad_norm) < abs(self._avqe_thresh):
            self._converged = 1
            self._final_energy = self._energies[-1]
            self._final_result = self._results[-1]
        else:
            self._converged = 0

    def get_num_ham_measurements(self):
        self._n_ham_measurements = 0
        for res in self._results:
            self._n_ham_measurements += res.nfev
        return self._n_ham_measurements

    def get_num_comut_measurements(self):
        self._n_comut_measurements = len(self._tamps) * len(self._pool)
        if self._use_analytic_grad:
            for m, res in enumerate(self._results):
                self._n_comut_measurements += res.njev * (m+1)

        return self._n_comut_measurements

    def get_final_energy(self, hit_max_avqe_iter=0):
        if hit_max_avqe_iter:
            print("\nADAPT-VQE at maximum number of iterations!")
            self._final_energy = self._energies[-1]
        else:
            return self._final_energy

    def get_final_result(self, hit_max_avqe_iter=0):
        if hit_max_avqe_iter:
            self._final_result = self._results[-1]
        else:
            return self._final_result
