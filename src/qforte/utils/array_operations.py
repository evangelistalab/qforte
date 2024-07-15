"""
array_operations.py
=================================================
A module for working with Scipy arrays
"""

import numpy as np
import scipy
import copy
from pytest import approx
from qforte import Computer


def number(a):
    # Get sum of bits in binary representation of a. (I.e., particle number.)
    sum_digits = 0
    while a > 0:
        sum_digits += a & 1
        a >>= 1
    return sum_digits


def spin(a):
    spin = 0
    b = copy.deepcopy(a)
    a >>= 1
    while a > 0:
        spin += a & 1
        a >>= 2
    while b > 0:
        spin -= b & 1
        b >>= 2
    return spin / 2


def sq_op_to_scipy(sq_op, N_qubits, N=None, Sz=None):
    # Function to convert a second-quantized operator to a sparse, complex matrix
    # Draws heavily from the sparse_tools module in OpenFermion

    X = scipy.sparse.csc_matrix(np.array([[0, 1 + 0j], [1 + 0j, 0]]))
    Y = scipy.sparse.csc_matrix(np.array([[0, 0 - 1j], [0 + 1j, 0]]))

    # Hilbert space dimension
    dim = 1 << N_qubits

    # Build all the annihilation operators
    annihilators = []
    zvec = np.ones((1))
    for i in range(0, N_qubits):
        ann = scipy.sparse.csc_matrix(0.5 * np.ones((1, 1)))
        big_Z = scipy.sparse.diags(zvec)
        big_id = scipy.sparse.identity(
            1 << (N_qubits - i - 1), dtype="complex", format="csc"
        )
        ann = scipy.sparse.kron(ann, big_id)
        ann = scipy.sparse.kron(ann, X + 1j * Y)
        ann = scipy.sparse.kron(ann, big_Z)
        zvec = np.kron(zvec, np.array([1, -1]))
        ann.eliminate_zeros()
        annihilators.append(ann)

    # Convert each excitation string to a sparse matrix and add them up.
    terms = sq_op.terms()
    vals = [[]]
    rows = [[]]
    cols = [[]]
    for term in terms:
        coeff = term[0]

        term_mat = scipy.sparse.identity(dim, dtype="complex", format="csc")
        for i in term[1]:
            term_mat = annihilators[i] @ term_mat
        for i in term[2]:
            term_mat = annihilators[i].getH() @ term_mat
        term_mat = term_mat.tocoo(copy=False)
        term_mat.eliminate_zeros()

        val = coeff * term_mat.data
        row, col = term_mat.nonzero()

        if N != None:
            accept = [
                i for i in range(len(val)) if number(row[i]) == number(col[i]) == N
            ]
            row = [row[i] for i in accept]
            col = [col[i] for i in accept]
            val = [val[i] for i in accept]

        if Sz != None:
            accept = [
                i
                for i in range(len(val))
                if spin(row[i])
                == approx(spin(col[i]), abs=1e-12)
                == approx(Sz, abs=1e-12)
            ]
            row = [row[i] for i in accept]
            col = [col[i] for i in accept]
            val = [val[i] for i in accept]

        vals.append(val)
        rows.append(row)
        cols.append(col)

    vals = np.concatenate(vals)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    arr = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(dim, dim))
    arr = arr.tocsc(copy=False)
    arr.eliminate_zeros()

    return scipy.sparse.csc_matrix(arr)


def build_effective_array(O, U, basis):
    """
    O is a qubit operator.
    U is a quantum circuit.
    basis is a list of Computer objects.
    This will return the matrix U'OU in the provided basis.
    """
    O_eff = np.zeros((len(basis), len(basis)), dtype="complex128")

    for i in range(len(basis)):
        basis[i].apply_circuit(U)

    for i, b_i in enumerate(basis):
        for j, b_j in enumerate(basis):
            Ob_j = Computer(b_j)
            Ob_j.apply_operator(O)
            O_eff[i, j] = np.vdot(b_i.get_coeff_vec(), Ob_j.get_coeff_vec())

    return O_eff
