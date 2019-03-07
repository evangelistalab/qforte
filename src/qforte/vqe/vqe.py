"""
A class for using an experiment to execute the variational quantum eigensolver
"""

import qforte
import numpy
import scipy
from scipy.optimize import minimize
from qforte.experiment import * 

class VQE(object):
    """
    VQE is a class that encompases the three componants of using the variational
    quantum eigensolver to optemize a parameterized wave function. Over a number
    of iterations the VQE: (1) prepares a quantum state on the quantum computer
    representing the wave function to be simulated, (2) evauates the energy by
    measurement, and (3) optemizes the the wave funciton by minimizing the energy
    via a gradient free algorithm such as nelder-mead simplex algroithm.
    VQE object constructor.
    :param n_qubits: the number of qubits for the quantum experiment.
    :param generator: the perameterized state preparation circuit.
    :param operator: the qubit operator to be measured.
    :param N_samples: the number of measurements made for each term in the operator
    :param many_preps: do a state preparation for every measurement (like a physical
        quantum computer would need to do).
    """

    def __init__(self, n_qubits, generator, operator, N_samples, many_preps = False, optemizer='nelder-mead'):
        self.n_qubits_ = n_qubits
        self.generator_ = generator
        self.operator_ = operator
        self.N_samples_ = N_samples
        self.many_preps_ = many_preps
        self.optemizer_ = optemizer
        self.experiment_ = qforte.Experiment(n_qubits, generator, operator, N_samples)

    """
    Optemizes the params of the generator by minimization of the
    'experimental_avg' function.
    """

    def do_vqe(self):

        #1 create an experiment (in constructor)
        #experiment = qforte.Experiment(self.n_qubits_, self.generator_, Self.operator_, Self.N_samples_)

        #2 assume the initial wfn is the HF wfn: set initial params such that this is the case.
        # TODO: check Nan's porting code and explore how the param vec should be organized (Nick)
        # need to read from openferm then perform troterization... (see pyquill for examples)
        params0 = []

        #3 run the optemize the wfn via minimization of the 'experimental_avg' function
        result = minimize(self.experiment_.experimental_avg, params0, method=self.optemizer_, options={'xtol': 1e-8, 'disp': True})

        #4 return the optemized params (result.x) and the lowest measured value,
        #as well as other useful information about the optemization.

        return result
