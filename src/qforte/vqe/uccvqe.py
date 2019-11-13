"""
uccvqe.py
====================================
A class for using an experiment to execute the variational quantum eigensolver
for a unitary coupled cluster ansatz
"""

import qforte
from qforte.experiment import *
from qforte.utils.transforms import *
from qforte.ucc.ucc_helpers import *
import numpy as np
import scipy
from scipy.optimize import minimize

class UCCVQE(object):
    """
    A class that encompases the three componants of using the variational
    quantum eigensolver to optemize a parameterized unitary CC wave function. Over a number
    of iterations the VQE: (1) prepares a quantum state on the quantum computer
    representing the wave function to be simulated, (2) evauates the energy by
    measurement, and (3) optemizes the the wave funciton by minimizing the energy.
    VQE object constructor.

    Attributes
    ----------
    _ref : list
        The set of 1s and 0s indicating the initial quantum state
    _Top : list of lists with tuple and float
        The cluster opterator T in a second quantized format, it is
        represented in the form
        [ [(p,q), t_pq], .... , [(p,q,s,r), t_pqrs], ... ]
        where p, q, r, s are idicies of normal ordered creation or anihilation
        operators.
    _operator : QuantumOperator
        The operator to be measured (usually the Hamiltonain), mapped to a
        qubit representation.
    _N_samples : int
        The number of times to measure each term in _operator.
    _alredy_anti_herm : bool
        Wether or not Top is passed already in an anti-hermetian form.
    _use_symmetric_amps : bopl
        Reduce the number of paramaters to be optemized by enforcing the
        symmetry of certain amplitudes. (Not yet functional).
    _many_preps : bool
        Use a new state preparation for every measurement.
        (Not yet functional).
    _optimizer : str
        The scipy optemizer to be used for the VQE.

    Methods
    -------
    unpack_args()
        Takes the second quantized operator and splits it into a list of
        amplitueds and a list of tuples containing normal ordered anihilator
        and creator indicies.
    repack_args(x, args)
        Takes a list of amplitudes and list of anihilator/creator tuples and
        creates a single list with both.
    expecation_val_func(x, *args)
        Calculates the energy expectatoin value with the state parameterized by
        the amplitues in x corresponding to excitations in *args.
    expecation_val_func_fast(x, *args)
        Calculates the energy expectatoin value without using tools from
        the Experiment class. The method is more efficient but unphysical
        for a quantum computer.
    do_vqe(fast=False, maxiter=20000)
        Runs the optemizer to mimimize the energy.
    get_result()
        Gets the output object from the scipy optemizer which contains information
        about the optemization (if it converhed, the number of iterations, final
        list of parameters, etc..).
    get_energy()
        Gets the minimum function value found by the minimizer.
    get_inital_guess_energy()
        Gets the energy calculated with the paramaters givin as an inital guess.
    """

    #TODO: Fix N_samples arg in Experiment class to only be take for finite measurement
    def __init__(self, ref, Top, operator, N_samples=100, alredy_anti_herm=False,
                 use_symmetric_amps = False, many_preps = False, optimizer='nelder-mead'):
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
        use_symmetric_amps : bopl
            Reduce the number of paramaters to be optemized by enforcing the
            symmetry of certain amplitudes. (Not yet functional).
        many_preps : bool
            Use a new state preparation for every measurement.
            (Not yet functional).
        optimizer : str
            The scipy optemizer to be used for the VQE.
        """
        #TODO(Nick): Elimenate getting info about nqubits in the 'len(ref)' fashion
        self._ref = ref
        self._nqubtis = len(ref)
        self._Top = Top
        self._operator = operator
        self._N_samples = N_samples
        self._alredy_anti_herm = alredy_anti_herm
        self._use_symmetric_amps = use_symmetric_amps
        self._many_preps = many_preps
        self._optimizer = optimizer

    def unpack_args(self):
        """Takes the second quantized operator and splits it into a list of
        amplitueds and a list of tuples containing normal ordered anihilator
        and creator indicies.

        Returns the x list of optemizable parameters and a tuple of static args,
        retrun format is NOT anti-hermitian
        """
        x = []
        args_lst = []
        for sq_op, amp in self._Top:
            x.append(amp)
            args_lst.append(sq_op)

        if(self._alredy_anti_herm):
            return x[::2], tuple(args_lst[::2])

        else:
            return x, tuple(args_lst)

    def repack_args(self, x, args):
        """Takes a list of amplitudes and list of anihilator/creator tuples and
        creates a single list with both.

        The return value sq_op is of the form
        [ [ (p,q), t_pq ], ..., [ (p,q,s,r), t_pqrs ], ... ].

        Parameters
        ----------
        x : list
            The list of coupled cluster amplitudes ordered by increasing excitation
            order [ t_pq, ..., t_pqrs, ..., t_pqrstu, ... ].
        args : list of tuples
            The list of tuples containing indicies of normal ordered creation/anihilation
            operators corresponindg to the amplitudes in x.
            The list is orgainized as
            [ (p,q), ..., (p,q,s,r), ..., (p,q,r,u,t,s), ... ].
        """
        sq_op = []
        if(self._use_symmetric_amps):
            raise ValueError('Simplification with symmetric amplitues in not yet avalible.')
            for i in range(len(x)):
                #NOTE: would be a good place to flip sign?
                #NOTE: will not currently support quintupple or higher order excitation
                sq_op.append([args[i], x[i]])

        else:
            for i in range(len(x)):
                #NOTE: would be a good place to flip sign?
                sq_op.append([args[i], x[i]])

        return sq_op

    def expecation_val_func(self, x, *args):
        """Calculates the energy expectatoin value without using tools from
        the Experiment class. The method is more efficient but unphysical
        for a quantum computer.

        Parameters
        ----------
        x : list
            The list of coupled cluster amplitudes ordered by increasing excitation
            order [ t_pq, ..., t_pqrs, ..., t_pqrstu, ... ].
        args : list of tuples
            The list of tuples containing indicies of normal ordered creation/anihilation
            operators corresponindg to the amplitudes in x.
            The list is orgainized as
            [ (p,q), ..., (p,q,s,r), ..., (p,q,r,u,t,s), ... ].
        """

        T_sq = self.repack_args(x, args)
        T_organizer = get_ucc_jw_organizer(T_sq, already_anti_herm=self._alredy_anti_herm)
        A = organizer_to_circuit(T_organizer)

        temp_op1 = qforte.QuantumOperator() # A temporary operator to multiply H by
        for t in A.terms():
            c, op = t
            phase =  -c
            temp_op1.add_term(phase, op)

        U, phase1 = qforte.trotterization.trotterize(temp_op1)
        cir = qforte.QuantumCircuit()
        for j in range(len(self._ref)):
            if self._ref[j] == 1:
                cir.add_gate(qforte.make_gate('X', j, j))

        cir.add_circuit(U)
        uccsd_exp = qforte.Experiment(len(self._ref), cir, self._operator, self._N_samples)
        params = [1.0]
        Energy = uccsd_exp.perfect_experimental_avg(params)
        #TODO: implement with finite measurement

        return Energy

    def expecation_val_func_fast(self, x, *args):
        """Calculates the energy expectatoin value without using tools from
        the Experiment class. The method is more efficient but unphysical
        for a quantum computer.

        Parameters
        ----------
        x : list
            The list of coupled cluster amplitudes ordered by increasing excitation
            order [ t_pq, ..., t_pqrs, ..., t_pqrstu, ... ].
        args : list of tuples
            The list of tuples containing indicies of normal ordered creation/anihilation
            operators corresponindg to the amplitudes in x.
            The list is orgainized as
            [ (p,q), ..., (p,q,s,r), ..., (p,q,r,u,t,s), ... ].
        """

        T_sq = self.repack_args(x, args)
        T_organizer = get_ucc_jw_organizer(T_sq, already_anti_herm=self._alredy_anti_herm)
        A = organizer_to_circuit(T_organizer)

        temp_op1 = qforte.QuantumOperator()
        for t in A.terms():
            c, op = t
            phase =  -c
            temp_op1.add_term(phase, op)

        U, phase1 = qforte.trotterization.trotterize(temp_op1)
        cir = qforte.QuantumCircuit()
        for j in range(len(self._ref)):
            if self._ref[j] == 1:
                cir.add_gate(qforte.make_gate('X', j, j))

        cir.add_circuit(U)
        QC = qforte.QuantumComputer(self._nqubtis)
        QC.apply_circuit(cir)
        QC.apply_constant(phase1)

        omega = np.asarray(QC.get_coeff_vec(), dtype=complex)
        Homega = np.zeros((2**self._nqubtis), dtype=complex)
        for k in range(len(self._operator.terms())):
            QCk = qforte.QuantumComputer(self._nqubtis)
            QCk.set_coeff_vec(QC.get_coeff_vec())

            if(self._operator.terms()[k][1] is not None):
                QCk.apply_circuit(self._operator.terms()[k][1])
            if(self._operator.terms()[k][0] is not None):
                QCk.apply_constant(self._operator.terms()[k][0])

            Homega = np.add(Homega, np.asarray(QCk.get_coeff_vec(), dtype=complex))

        Energy = np.vdot(omega, Homega)

        return np.real(Energy)

    def do_vqe(self, fast=False, maxiter=20000):
        """Runs the optemizer to mimimize the energy. Sets certain optemizer parameters
        internally.

        Parameters
        ----------
        fast : bool
            Wether or not to use the optemized but unphysical energy evaluation
            function.
        maxiter : int
            The maximum number of iterations for the scipy optemizer.
        """

        #TODO(Nick): make below options accessible from a higher level

        opts = {}
        opts['xatol'] = 1e-4
        opts['fatol'] = 1e-5
        opts['disp'] = True
        opts['maxiter'] = maxiter

        x0, sq_args = self.unpack_args()
        if(fast):
            self._inital_guess_energy = self.expecation_val_func_fast(x0, *sq_args)
            self._result = minimize(self.expecation_val_func_fast, x0,
                args=sq_args,  method=self._optimizer,
                options=opts)

        else:
            self._inital_guess_energy = self.expecation_val_func(x0, *sq_args)
            self._result = minimize(self.expecation_val_func, x0,
                args=sq_args,  method=self._optimizer,
                options=opts)

    def get_result(self):
        """Gets the output object from the scipy optemizer which contains information
        about the optemization (if it converhed, the number of iterations, final
        list of parameters, etc..).
        """
        return self._result

    def get_energy(self):
        """Gets the minimum function value found by the minimizer.
        """
        if(self._result.success):
            return self._result.fun
        else:
            print('\nresult:', self._result)
            return self._result.fun

    def get_inital_guess_energy(self):
        """Gets the energy calculated with the paramaters givin as an inital guess.
        """
        return self._inital_guess_energy
