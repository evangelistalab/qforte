"""
Tools for building and diagonalizing operators in subspaces obtained by methods like SA-ADAPT-VQE.
"""
import qforte
import numpy as np
from qforte.utils.compute_matrix_element import compute_operator_matrix_element

def build_effective_symmetric_operator(qb_op, Us):
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
        for j in range(i, dim):
            val = compute_operator_matrix_element(qb_op.num_qubits(), Us[j], Us[i], qb_op)            
            eff_op[i,j] = eff_op[j,i] = val

    return eff_op    

    
