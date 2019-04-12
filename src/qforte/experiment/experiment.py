"""
A class for running simulated quantum computing experiments, contains functions
that will return approximate expectation values.
"""

import qforte
import numpy

class Experiment(object):
    """
    Experimant is a class that exemplifies two quantum computational tasks:
    (1) state preparation from a 'generator' circuit which may or may not be
    parameterized, and (2) to measure operators to produce approximate
    expectaion values.

    Experiment object constructor.

    :param n_qubits: the number of qubits for the quantum experiment.

    :param generator: the perameterized state preparation circuit.

    :param operator: the qubit operator to be measured.

    :param N_samples: the number of measurements made for each term in the operator

    :param many_preps: do a state preparation for every measurement (like a physical
        quantum computer would need to do).

    """

    def __init__(self, n_qubits, n_elec, generator, operator, N_samples, RHF_ref = True, many_preps = False):
        self.n_qubits_ = n_qubits
        self.n_elec_ = n_elec
        self.generator_ = generator
        self.operator_ = operator
        self.N_samples_ = N_samples
        self.RHF_ref_ = RHF_ref
        self.many_preps_ = many_preps

    """
    Calculates the experimental average of the operator the Experiment object was initialized with.

    :param params: the list of parameters for the state preparation ansatz.

    """

    def experimental_avg(self, params):

        if(self.many_preps_==False):
            #1 initialize a quantum computer
            qc = qforte.QuantumComputer(self.n_qubits_)

            # set up for HF
            if(self.RHF_ref_):
                HFgen = qforte.QuantumCircuit()
                for n in range(self.n_elec_):
                    HFgen.add_gate(qforte.make_gate('X', n, n))

                qc.apply_circuit(HFgen)

            #2 build/update generator with params
            self.generator_.set_parameters(params)

            #3 apply generator (once if many_preps = False, N_sample times if)
            qc.apply_circuit(self.generator_)

            #4 check to see if operator is of mirror type and measure operator
            n_terms = len(self.operator_.terms())
            term_sum = 0.0

            if(self.operator_.get_is_mirror()==True):
                for k in range(n_terms):
                    measured = qc.measure_rotated_circuit(self.operator_.terms()[k][1], self.N_samples_)
                    term_sum += self.operator_.terms()[k][0] * sum(measured)

            elif(self.operator_.get_is_mirror()==False):
                for k in range(n_terms):
                    measured = qc.measure_circuit(self.operator_.terms()[k][1], self.N_samples_)
                    term_sum += self.operator_.terms()[k][0] * sum(measured)

            term_sum /= self.N_samples_

            return numpy.real(term_sum)

        elif(self.many_preps_==True):
            raise Exception('No support yet for measurement with multiple state preparations')

    #Have VQE function here? or make in a separate class?
