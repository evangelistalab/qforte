import qforte as qf


def compute_operator_matrix_element(n_qubit, U_bra, U_ket, QOp=None):
    """
    This function computes expressions of the form:

    <Psi_bra| QOp | Psi_ket> = <0| U_bra^dagger QOp U_ket |0>.

    Note that, if required, the U_bra and U_ket circuits should
    contain the X strings to convert the <0| and |0> states to
    the desired Slater determinants.

    Arguments
    =========

    n_qubit: int
        Number of qubits of the system.

    U_bra: Circuit object
        The quantum circuit that defines |Psi_bra> = U_bra |0>.
        The adjoint of U_bra is constructed by this function.

    U_ket: Circuit object
        The quantum circuit that defines |Psi_ket> = U_ket |0>.

    QOp: QubitOperator object or None
        The operator whose matrix element we are computing.
        If QOp is None, then the <Psi_bra|Psi_ket> overlap
        is computed.

    Returns
    =======

    The value of the desired matrix element.
    """

    comp = qf.Computer(n_qubit)
    comp.apply_circuit(U_ket)

    if QOp is not None:
        comp.apply_operator(QOp)
    comp.apply_circuit(U_bra.adjoint())

    return comp.get_coeff_vec()[0]
