"""
Functions for trotterization of qubit operators
"""

import qforte
import numpy


def trotterize(operator, trotter_number=1, trotter_order=1):

    """
    returns a circuit equivilant to an exponentiated QuantumOperator

    :param operator: (QuantumOperator) the operator or state preparation ansatz
    (represented as a sum of pauli terms) to be exponentiated

    :param trotter_number: (int) for an operator A with terms A_i, the trotter_number
    is the exponent (N) for to product of single term
    exponentals e^A ~ ( Product_i(e^(A_i/N)) )^N

    :param trotter_order: (int) the order of the troterization approximation, can be 1 or 2
    """

    if(trotter_order > 1) or (trotter_order <= 0):
        raise ValueError("trotterization currently only supports trotter order = 1")
    if(trotter_number > 1) or (trotter_number <= 0):
        raise ValueError("trotterization currently only supports trotter number = 1")

    total_phase = 1.0
    troterized_operator = qforte.QuantumCircuit()

    if (trotter_order == 1) and (trotter_number == 1):
        #loop over terms in operator
        for term in operator.terms():
            term_generator, phase = qforte.exponentiate_single_term(term[0],term[1])
            for gate in term_generator.gates():
                troterized_operator.add_gate(gate)
            total_phase *= phase

    return (troterized_operator, total_phase)
