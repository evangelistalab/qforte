"""
Tools for building and diagonalizing operators in subspaces obtained by methods like SA-ADAPT-VQE.
"""
import qforte
import numpy as np


def build_effective_operator(qb_op, Us):
    """
    qb_op is a qubit operator (e.g. a Hamiltonian, S^2, dipole operator)
    Us is a list of circuits to prepare a set of basis states.

    A dense np array will be constructed in the space of those basis states.
    
    TODO: Add a Hadamard test implementation for noise/gate count analysis
    """

    dim = len(Us)
    N_qb = qb_op.num_qubits()
    eff_op = np.zeros((dim, dim), dtype = "complex")
    for i in range(dim):
        myQC = qforte.Computer(N_qb)
        myQC.apply_circuit(Us[i])
        myQC.apply_operator(qb_op)
        sig = np.array(myQC.get_coeff_vec())
        for j in range(dim):
            myQC = qforte.Computer(N_qb)
            myQC.apply_circuit(Us[j])
            vec = np.array(myQC.get_coeff_vec())
            val = sig.conj().T@vec
            eff_op[i,j] = eff_op[j,i] = val

    return eff_op    

    
