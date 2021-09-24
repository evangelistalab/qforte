"""
Functions for performing qubit tapering based on the Z2 symmetries of the qubit Hamiltonian
"""

import qforte as qf
import numpy as np
from itertools import product

def find_Z2_symmetries(hamiltonian, qiskit_order = False):

    """
    This function computes the Z2 symmetries of the Hamiltonian, identifies the ID numbers of the qubits to
    be tapered off, provides the corresponding sign structures, and constructs the necessary unitary operators.
    It is based on the qubit tapering approach reported in https://arxiv.org/abs/1701.08213.

    Arguments:

    hamiltonian: QubitOperator
        The qubit Hamiltonian whose Z2 symmetries we want to find.

    qiskit_order: bool
        If True, the qubits to be tapered off will have the smallest id numbers possible, as is done in Qiskit.
        If False, the qubits to be tapered off will have the largest id numbers possible. This is the default,
        since it leads to less operations once the qubits are removed.
    """

    # The algorithm consists of the following steps:
    # 1) Convert qubit Hamiltonian to binary matrix, called the parity check matrix E
    # 2) Find the basis of the kernel of the parity check matrix, ker{E}
    # 3) Find maximal Abelian subgroup of the basis of ker{E}
    # 4) Construct unitary matrices that will be used in the qubit_tapering algorithm
    # 5) Compute the various signs corresponding to the eigenvalues of the sigma_x operators that will be removed

    # 1) Convert qubit Hamiltonian to binary matrix

    ## Obtain number of qubits and number of Pauli strings contained in qubit Hamiltonian
    n_qubits = hamiltonian.num_qubits()
    n_strings = len(hamiltonian.terms())

    ## Construct parity check matrix
    prt_chck_mtrx = np.zeros((n_strings, 2 * n_qubits), dtype=np.uint8)
    for i, string in enumerate(hamiltonian.terms()):
        for pauli in string[1].gates():
            XYZ_gate = pauli.gate_id()
            trgt = pauli.target()
            if XYZ_gate == 'Z':
                prt_chck_mtrx[i, n_qubits - trgt - 1] = 1
            elif XYZ_gate == 'X':
                prt_chck_mtrx[i, 2 * n_qubits - trgt - 1] = 1
            else:
                prt_chck_mtrx[i, n_qubits - trgt - 1] = 1
                prt_chck_mtrx[i, 2 * n_qubits - trgt - 1] = 1

    # 2) Find the basis of the kernel of the parity check matrix, ker{E}
    # See https://en.wikipedia.org/wiki/Kernel_(linear_algebra)#Computation_by_Gaussian_elimination

    ## Augment the (n_strings) × (2 * n_qubits) parity check matrix by the (2 * n_qubits) × (2 * n_qubits) identity matrix

    prt_chck_mtrx_aug = np.concatenate((prt_chck_mtrx, np.identity(2 * n_qubits, dtype=np.uint8)), axis=0)

    ## Find column echelon form (CEF) of augmented matix.

    idx = 0
    for row in range(n_strings):
        ### Find a non-zero row
        if np.any(prt_chck_mtrx_aug[row,idx:] == 1):
            ### Find positions of non-zero elements
            count = np.argwhere(prt_chck_mtrx_aug[row,idx:] == 1)
            column = int(count[0]) + idx
            ### Swap columns to bring first non-zero element in this row immediately to the right of that of the previous row
            prt_chck_mtrx_aug[:, [column , idx]] = prt_chck_mtrx_aug[:, [idx, column]]
            ### Eliminate remaining non-zero elements in the submatrix
            for i in range(1,count.shape[0]):
                column2 = int(count[i]) + idx
                prt_chck_mtrx_aug[:,column2] = np.bitwise_xor(prt_chck_mtrx_aug[:, idx], prt_chck_mtrx_aug[:,column2])
            ### decrease the number of columns that need to be checked
            idx += 1

    ## Find basis vectors of kernel by checking for zero columns in the CEF of the parity check matrix

    dim = 0
    basis = np.zeros((1, 2 * n_qubits), dtype=np.uint8)
    for column in range(2 * n_qubits):
        row = n_strings
        if not np.any(prt_chck_mtrx_aug[:row, column]):
            if dim == 0:
                basis = prt_chck_mtrx_aug[row:, column].reshape((1,2 * n_qubits))
                dim += 1
            else:
                basis = np.concatenate((basis, prt_chck_mtrx_aug[row:, column].reshape((1,2 * n_qubits))), axis=0)

    # 3) Find maximal Abelian subgroup of the basis of ker{E}
    # This is done using the symplectic bilinear form B that the binary vector space is equipped with:
    # B(v,w) = v * B * w.T,
    # where v and w are binary vectors and B is the matrix of the bilinear form.

    ## Construct matrix of bilinear form.

    top = np.concatenate((np.zeros((n_qubits, n_qubits), dtype=np.uint8), np.identity(n_qubits, dtype=np.uint8)), axis=1)
    bottom = np.concatenate((np.identity(n_qubits, dtype=np.uint8), np.zeros((n_qubits, n_qubits), dtype=np.uint8)), axis=1)
    blnr_frm_mtrx = np.concatenate((top, bottom), axis=0)

    ## Construct abln matrix.
    ## abln[i,j] contains the value of the bilinear form betweeen the ith and jth basis vectors of ker{E}
    ## A "0"/"1" in the ith row and jth column of abln means that the Pauli strings
    ## associated with the ith and jth binary vectors commute/anticommute.

    aux = np.matmul(blnr_frm_mtrx, basis.T)

    abln = np.zeros((basis.shape[0], basis.shape[0]), dtype=np.uint8)
    for row in range(basis.shape[0]):
        for column in range(basis.shape[0]):
            for idx in range(2*n_qubits):
                abln[row, column] = (abln[row, column] + basis[row, idx] * aux[idx, column]) % 2

    ## Find the maximal abelian subgroup iteratively. This is done as follows:
    ## 1) Find non-zero row of abln with smallest Hamming weight.
    ##    The corresponding basis vector e commutes with most other basis vectors.
    ## 2) Find basis vectors that do not commute with e and remove them.
    ## 3) Reconstruct abln matrix with remaining basis vectors.
    ## 4) If abln == 0 exit otherwise go to step 1) and repeat.

    basis_tmp = basis

    while not np.all(abln == 0):

        ### Maximum possible Hamming weight of a given binary vector
        hmng = 2 * n_qubits

        idx = 0
        for row in range(abln.shape[0]):
            ### Find the basis vector e that commutes with most of the remaining basis vectors
            hmng_tmp = np.sum(abln[row, :])
            if hmng_tmp > 0 and hmng > hmng_tmp:
                hmng = hmng_tmp
                idx = row

        ### Identify basis vectors that do not commute with e and eliminate them.
        count = np.argwhere(abln[idx,:] == 1)
        basis_tmp = np.delete(basis_tmp, count, axis=0)

        ### Construct abln matrix for the next iteration.

        aux = np.matmul(blnr_frm_mtrx, basis_tmp.T)

        abln = np.zeros((basis_tmp.shape[0], basis_tmp.shape[0]), dtype=np.uint8)
        for row in range(basis_tmp.shape[0]):
            for column in range(basis_tmp.shape[0]):
                for idx in range(2*n_qubits):
                    abln[row, column] = (abln[row, column] + basis_tmp[row, idx] * aux[idx, column]) % 2

    tau = basis_tmp

    #### This step is not necessary. It transforms the generators of the symmetry group in such a way that the
    #### removed qubits have the smallest/largest possbile identity numbers. A similar modification is made
    #### at the "sigma_x selection" part of the algorithm below. The "smallest" case produces identical resutls
    #### with those obtained using Qiskit, after taking into account that in Qiskit Slater determinants are
    #### represented by strings of alpha orbitals followed by the beta ones.
    idx = 0
    lst = []
    if qiskit_order:
        _range = list(reversed(range(n_qubits, 2 * n_qubits)))
    else:
        _range = list(range(n_qubits, 2 * n_qubits))
    for column in _range:
        count = np.argwhere(tau[:, column] == 1)
        if count.shape[0] == 1:
            if count[0] not in lst:
                lst.append(int(count[0]))
                idx += 1
        else:
            for i in range(count.shape[0]):
                if count[i] not in lst:
                    lst.append(int(count[i]))
                    for el in filter(lambda el: el not in count[i], count):
                        tau[int(el), :] = np.bitwise_xor(tau[int(el), :], tau[int(count[i]), :])
                    idx += 1
                    break
        if idx == tau.shape[0]:
            break
    ####

    # 4) Construct unitary matrix that will transform Hamiltonian

    ## Translate binary vectors back to Pauli strings to obtain the generators of the symmetry group
    gnrtrs = []
    for i in range(tau.shape[0]):
        pauli = qf.Circuit()
        for j in reversed(range(n_qubits)):
            if tau[i, j] == 1 and tau[i, j + n_qubits] == 1:
                pauli.add_gate(qf.gate('Y', n_qubits - j - 1))
            elif tau[i, j] == 1:
                pauli.add_gate(qf.gate('X', n_qubits - j - 1))
            elif tau[i, j + n_qubits] == 1:
                pauli.add_gate(qf.gate('Z', n_qubits - j - 1))
        gnrtrs.append(pauli)

    ## Find which sigma_x gate will be paired with which generator
    sigma_x = np.zeros((tau.shape[0]), dtype=np.uint16)

    if qiskit_order:
        _range = list(range(n_qubits))
    else:
        _range = list(reversed(range(n_qubits)))
    for i in _range:
        if np.argwhere(tau[:,n_qubits + i] == 1).shape[0] == 1:
            sigma_x[np.argwhere(tau[:,n_qubits + i] == 1)] = n_qubits - i - 1

    ## Reorder sigma_x operators in descending order of qubit identity. Do the same with the generators for consistency.

    reorder = np.argsort(sigma_x)
    sigma_x = sigma_x[reorder][::-1]
    gnrtrs = [gnrtrs[i] for i in reorder[::-1]]

    ## Construct individual unitary matrices
    untrs = []

    for idx, generator in enumerate(gnrtrs):
        circ = qf.Circuit()
        circ.add_gate(qf.gate('X', sigma_x[idx]))
        oprtr = qf.QubitOperator()
        oprtr.add_term(1/np.sqrt(2), generator)
        oprtr.add_term(1/np.sqrt(2), circ)
        untrs.append(oprtr)

    ## Construct final unitary matrix
    unitary = qf.QubitOperator()
    unitary.add_op(untrs[0])
    for op in untrs[1:]:
        unitary.operator_product(op, True, True)

    # 5) Compute the various signs corresponding to the eigenvalues of the sigma_x operators that will be removed

    signs = list(product([1, -1], repeat=len(sigma_x)))

    return gnrtrs, sigma_x, untrs, unitary, signs

def qubit_tapering_oprtr(n_qubits, tapered_qubits, sign, operator, unitary):

    """
    This function uses the unitary matrix obtained using find_Z2_symmetries to transform a given operator.
    After the unitary transformation, the resulting operator acts on the qubits to be tapered off by at most
    a single sigma_x operator. The sigma_x gates acting on qubits to be tapered off are replaced by their
    +/- 1 eigenvalues, which are provided by the user.

    Arguments

    n_qubits: integer
        Initial number of qubits.

    tapered_qubits: list of integers
        The ID numbers of the qubits to be tapered off.

    sign: list of +/- 1.
        The desired sign structure corresponding to the eigenvalues of the sigma_x operators acting on
        the qubits to be tapered off.

    operator: QubitOperator
        The operator that we wish to transform. For example, the qubit Hamiltonian or the UCC ansatz.

    unitary: QubitOperator
        The unitary operator that will perform the dersired transformation.
    """

    # Transform the given operator using the fact that the unitary matrix is also Hermitian

    unitary_tmp = qf.QubitOperator()
    unitary_tmp.add_op(unitary)
    operator.operator_product(unitary_tmp, True, True)
    unitary_tmp.operator_product(operator, True, True)
    operator = unitary_tmp

    # Create a QForte QubitOperator object that will store the tapered operator
    operator_tapered = qf.QubitOperator()

    coeff_tmp = []
    for counter, term in enumerate(operator.terms()):
        # Create circuit that will store the pauli string with gates corresponding to qubits to be tapered off removed
        circ = qf.Circuit()
        # List that will store the coefficients modified according to the sign structure provided by the user
        coeff_tmp.append(term[0])
        for pauli in term[1].gates():
            if any(qubit == pauli.target() for qubit in tapered_qubits):
                coeff_tmp[counter] *= sign[int(np.argwhere(tapered_qubits == pauli.target()))]
            else:
                # Adjust the qubit ID of a given gate based on the number of gates that were removed before it
                idx = np.count_nonzero(pauli.target() > tapered_qubits)
                target = pauli.target() - idx
                gate = pauli.gate_id()
                circ.add_gate(qf.gate(gate, target))

        operator_tapered.add_term(term[0], circ)

    # Replace the coefficients with the ones having the requested sign structure
    operator_tapered.set_coeffs(coeff_tmp)
    # Simplify the entire operator
    operator_tapered.simplify(True)

    return operator_tapered

def qubit_tapering_ref(tapered_qubits, ref):

    """
    This function removes the designated qubits from a given reference.

    Arguments

    tapered_qubits: list of integers
        The ID numbers of the qubits to be tapered off.

    ref: list of integers
        A bit string representing the reference Slater determinant.
    """

    for i in tapered_qubits:
        del ref[i]

    return ref
