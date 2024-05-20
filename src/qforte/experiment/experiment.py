"""
A class for running simulated quantum computing experiments, contains functions
that will return approximate expectation values.
"""

import qforte
import numpy


class Experiment(object):
    def __init__(
        self, n_qubits, generator, operator, N_samples, prepare_each_time=False
    ):
        """
        Experiment is a class that exemplifies two quantum computational tasks:
        (1) state preparation from a 'generator' circuit which may or may not be
        parameterized, and (2) to measure operators to produce approximate
        expectation values.

        Experiment object constructor

        :param n_qubits: (int) the number of qubits for the quantum experiment.

        :param generator: (Circuit) the parameterized state preparation circuit.

        :param operator: (QubitOperator) the qubit operator to be measured.

        :param N_samples: (int) the number of measurements made for each term in the operator

        :param prepare_each_time: (bool) do a state preparation for every measurement (like a physical
            quantum computer would need to do).

        """

        self.n_qubits_ = n_qubits
        self.generator_ = generator
        self.operator_ = operator
        self.N_samples_ = N_samples
        self.prepare_each_time_ = prepare_each_time

    def experimental_avg(self, params=[]):
        """
        calculates the experimental average of the operator the Experiment object was initialized with

        :param params: (list) the list of parameters for the state preparation ansatz.

        """

        if self.prepare_each_time_ == False:
            # 1 initialize a quantum computer
            qc = qforte.Computer(self.n_qubits_)

            # 2 build/update generator with params
            # self.generator_.set_parameters(params)

            # 3 apply generator (once if prepare_each_time = False, N_sample times if)
            qc.apply_circuit(self.generator_)

            # 4 measure operator
            n_terms = len(self.operator_.terms())
            term_sum = 0.0

            for k in range(n_terms):
                measured = qc.measure_circuit(
                    self.operator_.terms()[k][1], self.N_samples_
                )
                term_sum += self.operator_.terms()[k][0] * sum(measured)

            term_sum /= self.N_samples_

            return numpy.real(term_sum)

        elif self.prepare_each_time_ == True:
            raise Exception(
                "No support yet for measurement with multiple state preparations"
            )

    def perfect_experimental_avg(self, params=[]):
        """
        calculates the exact experimental result of the operator the Experiment object was initialized with

        :param params: (list) the list of parameters for the state preparation ansatz.

        """

        if self.prepare_each_time_ == False:
            # 1 initialize a quantum computer
            qc = qforte.Computer(self.n_qubits_)

            # 2 build/update generator with params
            # self.generator_.set_parameters(params)

            # 3 apply generator (once if prepare_each_time = False, N_sample times if)
            qc.apply_circuit(self.generator_)

            # 4 measure operator
            n_terms = len(self.operator_.terms())
            term_sum = 0.0

            for k in range(n_terms):
                term_sum += self.operator_.terms()[k][0] * qc.perfect_measure_circuit(
                    self.operator_.terms()[k][1]
                )

            return numpy.real(term_sum)

        elif self.prepare_each_time_ == True:
            raise Exception(
                "No support yet for measurement with multiple state preparations"
            )

    # Have VQE function here? or make in a separate class?
