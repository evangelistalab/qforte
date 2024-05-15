"""
Functions for constructing the operators representing the square of the total
spin (S^2), the projection of the total spin on the z axis (S_z), and the
total spin ladder operators
"""

import qforte as qf


def total_spin_squared(n_qubits):
    """
    This function constructs the operator representing the square of the total spin
    in the qubit basis (assuming Jordan-Wigner encoding).

    Arguments
    =========

    n_qubits: int
        Number of qubits (spin-orbitals) of the system

    Returns
    =======

    q_Sigma_squared: QubitOperator
        S^2 operator in the qubit basis, using Jordan-Wigner encoding
    """

    q_Sigma_z = total_spin_z(n_qubits)
    q_Sigma_minus = total_spin_lowering(n_qubits)
    q_Sigma_plus = total_spin_raising(n_qubits)

    q_Sigma_squared = qf.QubitOperator()
    q_Sigma_z_squared = qf.QubitOperator()

    q_Sigma_squared.add_op(q_Sigma_minus)
    q_Sigma_squared.operator_product(q_Sigma_plus, True, True)
    q_Sigma_squared.add_op(q_Sigma_z)
    q_Sigma_z_squared.add_op(q_Sigma_z)
    q_Sigma_z_squared.operator_product(q_Sigma_z, True, True)
    q_Sigma_squared.add_op(q_Sigma_z_squared)

    q_Sigma_squared.simplify(True)

    return q_Sigma_squared


def total_spin_z(n_qubits):
    """
    This function constructs the operator representing the projection of the
    total spin on the z axis in the qubit basis (assuming Jordan-Wigner encoding)

    Arguments
    =========

    n_qubits: int
        Number of qubits (spin-orbitals) of the system

    Returns
    =======

    sq_Sigma_z.jw_transform(): QubitOperator
        Sz operator in the qubit basis, using Jordan-Wigner encoding
    """

    sq_Sigma_z = qf.SQOperator()

    for i in range(0, n_qubits, 2):
        sq_Sigma_z.add(0.5, [i], [i])
        sq_Sigma_z.add(-0.5, [i + 1], [i + 1])

    return sq_Sigma_z.jw_transform()


def total_spin_lowering(n_qubits):
    """
    This function constructs the lowering operator of the total spin
    in the qubit basis (assuming Jordan-Wigner encoding)

    Arguments
    =========

    n_qubits: int
        Number of qubits (spin-orbitals) of the system

    Returns
    =======

    sq_Sigma_minus.jw_transform(): QubitOperator
        S- operator in the qubit basis, using Jordan Wigner encoding
    """

    sq_Sigma_minus = qf.SQOperator()

    for i in range(0, n_qubits, 2):
        sq_Sigma_minus.add(1, [i + 1], [i])

    return sq_Sigma_minus.jw_transform()


def total_spin_raising(n_qubits):
    """
    This function constructs the raising operator of the total spin,
    in the qubit basis (assuming Jordan-Wigner encoding)

    Arguments
    =========

    n_qubits: int
        Number of qubits (spin-orbitals) of the system

    Returns
    =======

    sq_Sigma_plus.jw_transform(): QubitOperator
        S+ operator in the qubit basis, using Jordan Wigner encoding
    """

    sq_Sigma_plus = qf.SQOperator()

    for i in range(0, n_qubits, 2):
        sq_Sigma_plus.add(1, [i], [i + 1])

    return sq_Sigma_plus.jw_transform()
