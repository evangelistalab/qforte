"""
A class for using an experiment to execute the variational quantum eigensolver
"""

import qforte
from qforte.experiment import *
from qforte.utils.transforms import *
from qforte.ucc.ucc_helpers import *
import scipy
from scipy.optimize import minimize

class UCCVQE(object):
    """
    UCCVQE is a class that encompases the three componants of using the variational
    quantum eigensolver to optemize a parameterized unitary CC wave function. Over a number
    of iterations the VQE: (1) prepares a quantum state on the quantum computer
    representing the wave function to be simulated, (2) evauates the energy by
    measurement, and (3) optemizes the the wave funciton by minimizing the energy
    via a gradient free algorithm such as nelder-mead simplex algroithm.
    VQE object constructor.
    :param n_qubits: the number of qubits for the quantum experiment.
    :param operator: the qubit operator to be measured.
    :param N_samples: the number of measurements made for each term in the operator
    :param many_preps: do a state preparation for every measurement (like a physical
        quantum computer would need to do).
    """

    def __init__(self, ref, Top, operator, N_samples, alredy_anti_herm=False , many_preps = False, optimizer='nelder-mead'):
        self._ref = ref
        self._Top = Top
        self._operator = operator
        self._N_samples = N_samples
        self._alredy_anti_herm = alredy_anti_herm
        self._many_preps = many_preps
        self._optimizer = optimizer


    """
    Optemizes the params of the generator by minimization of the
    'experimental_avg' function.
    """

    def unpack_args(self):

        x = []
        args_lst = []
        for sq_op, amp in self._Top:
            x.append(amp)
            args_lst.append(sq_op)

        # Returns the x vector of optemizable parameters and a tuple of static args,
        # Retrun format is NOT anti-hermitian
        if(self._alredy_anti_herm):
            return x[::2], tuple(args_lst[::2])

        else:
            return x, tuple(args_lst)

    def repack_args(self, x, args):

        # assumes that x and args are NOT anti-hermetian already
        # sq_op => [ [ (i,j), t_ij ], [], ... ]
        # x => [t_ij, ...]
        # args => [(i,j), ...]

        sq_op = []
        for i in range(len(x)):
            #NOTE: would be a good place to flip sign?
            sq_op.append([args[i], x[i]])

        return sq_op

    def expecation_val_func(self, x, *args):

        # Unpack arguments
        T_sq = self.repack_args(x, args)
        T_organizer = get_ucc_jw_organizer(T_sq, already_anti_herm=self._alredy_anti_herm)
        A = organizer_to_circuit(T_organizer)

        temp_op1 = qforte.QuantumOperator() # A temporary operator to multiply H by
        for t in A.terms():
            c, op = t
            phase =  -c
            temp_op1.add_term(phase, op)

        U, phase1 = qforte.trotterization.trotterize(temp_op1)

        # Prep the reference state
        cir = qforte.QuantumCircuit()
        for j in range(len(self._ref)):
            if self._ref[j] == 1:
                cir.add_gate(qforte.make_gate('X', j, j))

        # Add unitary coupled cluster operator
        cir.add_circuit(U)

        # Compute energy via measurement
        uccsd_exp = qforte.Experiment(len(self._ref), cir, self._operator, 100)
        params = [1.0]
        Energy = uccsd_exp.perfect_experimental_avg(params)

        return Energy

    def do_vqe(self, maxiter=20000):

        # Set options dict
        opts = {}
        opts['xtol'] = 1e-4
        opts['ftol'] = 1e-5
        opts['disp'] = True
        opts['maxiter'] = maxiter
        # opts['maxfev'] = 1

        x0, sq_args = self.unpack_args()
        self._inital_guess_energy = self.expecation_val_func(x0, *sq_args)
        self._result = minimize(self.expecation_val_func, x0,
            args=sq_args,  method=self._optimizer,
            options=opts)

    def get_result(self):
        return self._result

    def get_energy(self):
        if(self._result.success):
            return self._result.fun
        else:
            print('\nresult:', self._result)
            # raise ValueError('Minimum energy invalid, optemization did not converge')
            return self._result.fun

    def get_inital_guess_energy(self):
        return self._inital_guess_energy
