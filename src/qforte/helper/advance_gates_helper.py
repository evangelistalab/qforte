import qforte

"""
This helper generates various advanced gates that could be
used to test quantum algorithms and ideas

The reference for those efficient decomposition is:
Nengkun Yu, Mingsheng Ying, arXiv:1301.3727 (2013)

"""


def Toffoli(i, j, k):
    """
    builds a circuit to simulate a three-qubit Toffoli gate
    (Control-Control-NOT, CCNOT gate).

    :param i: control qubit 1
    :param j: control qubit 2
    :param k: target qubit
    """

    T1 = qforte.gate("T", i, i)
    T2 = qforte.gate("T", j, j)
    T3 = qforte.gate("T", k, k)
    C12 = qforte.gate("cX", j, i)
    C13 = qforte.gate("cX", k, i)
    C23 = qforte.gate("cX", k, j)
    H3 = qforte.gate("H", k, k)

    T_circ = qforte.Circuit()
    T_circ.add(H3)
    T_circ.add(C23)
    T_circ.add(T3.adjoint())
    T_circ.add(C13)
    T_circ.add(T3)
    T_circ.add(C23)
    T_circ.add(T3.adjoint())
    T_circ.add(C13)
    T_circ.add(T2)
    T_circ.add(T3)
    T_circ.add(C12)
    T_circ.add(H3)
    T_circ.add(T1)
    T_circ.add(T2.adjoint())
    T_circ.add(C12)

    return T_circ


def Fredkin(i, j, k):
    """
    builds a circuit to simulate a three-qubit Fredkin gate
    (Controled-SWAP, CSWAP gate).

    :param i: control qubit 1
    :param j: swap qubit 1
    :param k: swap qubit 2
    """

    C12 = qforte.gate("cX", j, i)
    C32 = qforte.gate("cX", j, k)
    CV23 = qforte.gate("cV", k, j)
    CV13 = qforte.gate("cV", k, i)

    F_circ = qforte.Circuit()
    F_circ.add(C32)
    F_circ.add(CV23)
    F_circ.add(CV13)
    F_circ.add(C12)
    F_circ.add(CV23.adjoint())
    F_circ.add(C32)
    F_circ.add(C12)

    return F_circ
