import qforte

"""
This helper generates various advanced gates that could be
used to test quantum algorithms and ideas

The reference for those efficient decomposition is:
Nengkun Yu, Mingsheng Ying, arXiv:1301.3727 (2013)

"""

def Toffoli(i,j,k):

    """
    builds a circuit to simulate a three-qubit Toffoli gate
    (Control-Control-NOT, CCNOT gate).

    :param i: control qubit 1
    :param j: control qubit 2
    :param k: target qubit
    """

    T1 = qforte.make_gate('T', i, i)
    T2 = qforte.make_gate('T', j, j)
    T3 = qforte.make_gate('T', k, k)
    C12 = qforte.make_gate('cX', j, i)
    C13 = qforte.make_gate('cX', k, i)
    C23 = qforte.make_gate('cX', k, j)
    H3 = qforte.make_gate('H', k, k)

    T_circ = qforte.QuantumCircuit()
    T_circ.add_gate(H3)
    T_circ.add_gate(C23)
    T_circ.add_gate(T3.adjoint())
    T_circ.add_gate(C13)
    T_circ.add_gate(T3)
    T_circ.add_gate(C23)
    T_circ.add_gate(T3.adjoint())
    T_circ.add_gate(C13)
    T_circ.add_gate(T2)
    T_circ.add_gate(T3)
    T_circ.add_gate(C12)
    T_circ.add_gate(H3)
    T_circ.add_gate(T1)
    T_circ.add_gate(T2.adjoint())
    T_circ.add_gate(C12)

    return T_circ

def Fredkin(i,j,k):

    """
    builds a circuit to simulate a three-qubit Fredkin gate
    (Controled-SWAP, CSWAP gate).

    :param i: control qubit 1
    :param j: swap qubit 1
    :param k: swap qubit 2
    """

    C12 = qforte.make_gate('cX', j, i)
    C32 = qforte.make_gate('cX', j, k)
    CV23 = qforte.make_gate('cV', k, j)
    CV13 = qforte.make_gate('cV', k, i)

    F_circ = qforte.QuantumCircuit()
    F_circ.add_gate(C32)
    F_circ.add_gate(CV23)
    F_circ.add_gate(CV13)
    F_circ.add_gate(C12)
    F_circ.add_gate(CV23.adjoint())
    F_circ.add_gate(C32)
    F_circ.add_gate(C12)

    return F_circ
