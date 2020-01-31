"""
Functions for trotterization of qubit operators
"""

import qforte
import numpy
import copy


def trotterize(operator, trotter_number=1, trotter_order=1):

    """
    returns a circuit equivilant to an exponentiated QuantumOperator

    :param operator: (QuantumOperator) the operator or state preparation ansatz
    (represented as a sum of pauli terms) to be exponentiated

    :param trotter_number: (int) for an operator A with terms A_i, the trotter_number
    is the exponent (N) for to product of single term
    exponentals e^A ~ ( Product_i(e^(A_i/N)) )^N

    :param trotter_number: (int) the order of the troterization approximation, can be 1 or 2
    """

    total_phase = 1.0
    troterized_operator = qforte.QuantumCircuit()

    if (trotter_number == 1) and (trotter_order == 1):
        #loop over terms in operator
        for term in operator.terms():
            term_generator, phase = qforte.exponentiate_single_term(term[0],term[1])
            for gate in term_generator.gates():
                troterized_operator.add_gate(gate)
            total_phase *= phase


    else:
        ho_op = qforte.QuantumOperator()
        for k in range(1, trotter_number+1):
            for term in operator.terms():
                ho_op.add_term( term[0] / float(trotter_number) , term[1])

        for term in ho_op.terms():
            term_generator, phase = qforte.exponentiate_single_term(term[0],term[1])
            for gate in term_generator.gates():
                troterized_operator.add_gate(gate)
            total_phase *= phase

    return (troterized_operator, total_phase)

def trotterize_w_cRz(operator, ancilla_qubit_idx, Use_open_cRz=False, trotter_number=1, trotter_order=1):

    """
    returns a circuit equivilant to an exponentiated QuantumOperator in which each term
    in the trotterization exp(-i * theta_k ) only acts on the register if the ancilla
    qubit is in the |1> state.

    :param operator: (QuantumOperator) the operator or state preparation ansatz
    (represented as a sum of pauli terms) to be exponentiated

    :param ancilla_qubit_idx: (int) the index of the ancilla qubit

    :param Use_open_cRz: (bool) uses an open controlled Rz gate in exponentiation
    (see Fig. 11 on page 185 of Nielson and Chung's
    "Quantum Computation and Quantum Informatoin 10th Aniversary Ed.")

    :param trotter_number: (int) for an operator A with terms A_i, the trotter_number
    is the exponent (N) for to product of single term
    exponentals e^A ~ ( Product_i(e^(A_i/N)) )^N

    :param trotter_order: (int) the order of the troterization approximation, can be 1 or 2
    """

    total_phase = 1.0
    troterized_operator = qforte.QuantumCircuit()

    if (trotter_number == 1) and (trotter_order == 1):
        #loop over terms in operator
        if(Use_open_cRz):
            for term in operator.terms():
                term_generator, phase = qforte.exponentiate_single_term(term[0],term[1], Use_cRz=True, ancilla_idx=ancilla_qubit_idx, Use_open_cRz=True)
                for gate in term_generator.gates():
                    troterized_operator.add_gate(gate)
                total_phase *= phase
        else:
            for term in operator.terms():
                term_generator, phase = qforte.exponentiate_single_term(term[0],term[1], Use_cRz=True, ancilla_idx=ancilla_qubit_idx)
                for gate in term_generator.gates():
                    troterized_operator.add_gate(gate)
                total_phase *= phase

    else:
        if(trotter_order > 1):
            raise NotImplementedError("Higher order trotterization is not yet implemented.")
        ho_op = qforte.QuantumOperator()
        for k in range(1, trotter_number+1):
            k = float(k)
            for term in operator.terms():
                ho_op.add_term( term[0] / float(trotter_number) , term[1])

        if(Use_open_cRz):
            for term in ho_op.terms():
                term_generator, phase = qforte.exponentiate_single_term(term[0],term[1], Use_cRz=True, ancilla_idx=ancilla_qubit_idx, Use_open_cRz=True)
                for gate in term_generator.gates():
                    troterized_operator.add_gate(gate)
                total_phase *= phase
        else:
            for term in ho_op.terms():
                term_generator, phase = qforte.exponentiate_single_term(term[0],term[1], Use_cRz=True, ancilla_idx=ancilla_qubit_idx)
                for gate in term_generator.gates():
                    troterized_operator.add_gate(gate)
                total_phase *= phase

    return (troterized_operator, total_phase)
