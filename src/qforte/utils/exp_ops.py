"""
Utility functions for handeling exponentials of QubitOperators
"""

import qforte
import numpy as np

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm, expm_multiply


def get_scipy_csc_from_op(Hop, factor):
    nqubits = Hop.num_qubits()
    nbasis = 2**nqubits
    sp_mat = Hop.sparse_matrix(nqubits)

    Ivals = []
    Jvals = []
    data = []

    for I in sp_mat.to_vec_map():
        for J in sp_mat.to_vec_map()[I].to_map():
            Ivals.append(I)
            Jvals.append(J)
            data.append(factor * sp_mat.to_vec_map()[I].to_map()[J])

    return csc_matrix((data, (Ivals, Jvals)), shape=(nbasis, nbasis), dtype=complex)


def apply_time_evolution_op(qc, Hcsc, tn, nstates):
    qc_vec = np.array(qc.get_coeff_vec())

    return expm_multiply(Hcsc, qc_vec, start=0.0, stop=tn, num=nstates, endpoint=True)
