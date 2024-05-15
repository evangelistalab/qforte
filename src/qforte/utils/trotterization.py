"""
Functions for trotterization of qubit operators
"""

import qforte
import numpy
import copy


def trotterize(operator, factor=1.0, trotter_number=1, trotter_order=1):
    """
    returns a circuit equivalent to an exponentiated QubitOperator

    :param operator: (QubitOperator) the operator or state preparation ansatz
    (represented as a sum of pauli terms) to be exponentiated

    :param trotter_number: (int) for an operator A with terms A_i, the trotter_number
    is the exponent (N) for to product of single term
    exponentals e^A ~ ( Product_i(e^(A_i/N)) )^N

    :param trotter_number: (int) the order of the trotterization approximation, can be 1 or 2
    """

    total_phase = 1.0
    trotterized_operator = qforte.Circuit()

    if (trotter_number == 1) and (trotter_order == 1):
        # loop over terms in operator
        for term in operator.terms():
            term_generator, phase = qforte.exponentiate_pauli_string(
                factor * term[0], term[1]
            )
            for gate in term_generator.gates():
                trotterized_operator.add(gate)
            total_phase *= phase

    else:
        if trotter_order > 1:
            raise NotImplementedError(
                "Higher order trotterization is not yet implemented."
            )

        ho_op = qforte.QubitOperator()

        for k in range(1, trotter_number + 1):
            for term in operator.terms():
                ho_op.add(factor * term[0] / float(trotter_number), term[1])

        for trot_term in ho_op.terms():
            term_generator, phase = qforte.exponentiate_pauli_string(
                trot_term[0], trot_term[1]
            )
            for gate in term_generator.gates():
                trotterized_operator.add(gate)
            total_phase *= phase

    return (trotterized_operator, total_phase)


def trotterize_w_cRz(
    operator,
    ancilla_qubit_idx,
    factor=1.0,
    Use_open_cRz=False,
    trotter_number=1,
    trotter_order=1,
):
    """
    Returns a circuit equivalent to an exponentiated QubitOperator in which each term
    in the trotterization exp(-i * theta_k ) only acts on the register if the ancilla
    qubit is in the |1> state.

    :param operator: (QubitOperator) the operator or state preparation ansatz
    (represented as a sum of pauli terms) to be exponentiated

    :param ancilla_qubit_idx: (int) the index of the ancilla qubit

    :param Use_open_cRz: (bool) uses an open controlled Rz gate in exponentiation
    (see Fig. 11 on page 185 of Nielson and Chung's
    "Quantum Computation and Quantum Informatoin 10th Aniversary Ed.")

    :param trotter_number: (int) for an operator A with terms A_i, the trotter_number
    is the exponent (N) for to product of single term
    exponentals e^A ~ ( Product_i(e^(A_i/N)) )^N

    :param trotter_order: (int) the order of the trotterization approximation, can be 1 or 2
    """

    total_phase = 1.0
    trotterized_operator = qforte.Circuit()

    if (trotter_number == 1) and (trotter_order == 1):
        for term in operator.terms():
            term_generator, phase = qforte.exponentiate_pauli_string(
                factor * term[0],
                term[1],
                Use_cRz=True,
                ancilla_idx=ancilla_qubit_idx,
                Use_open_cRz=Use_open_cRz,
            )
            for gate in term_generator.gates():
                trotterized_operator.add(gate)
            total_phase *= phase

    else:
        if trotter_order > 1:
            raise NotImplementedError(
                "Higher order trotterization is not yet implemented."
            )
        ho_op = qforte.QubitOperator()
        for k in range(1, trotter_number + 1):
            k = float(k)
            for term in operator.terms():
                ho_op.add(factor * term[0] / float(trotter_number), term[1])

        for trot_term in ho_op.terms():
            term_generator, phase = qforte.exponentiate_pauli_string(
                trot_term[0],
                trot_term[1],
                Use_cRz=True,
                ancilla_idx=ancilla_qubit_idx,
                Use_open_cRz=Use_open_cRz,
            )
            for gate in term_generator.gates():
                trotterized_operator.add(gate)
            total_phase *= phase

    return (trotterized_operator, total_phase)
