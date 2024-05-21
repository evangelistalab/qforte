"""
Functions for performing qubit tapering based on the Z2 symmetries of the qubit Hamiltonian
"""

import qforte as qf
import numpy as np
from itertools import product
from copy import deepcopy


def find_Z2_symmetries(hamiltonian, taper_from_least=True, debug=False):
    """
    This function computes the Z2 symmetries of the qubit Hamiltonian, identifies the ID numbers of the qubits to
    be tapered off, and constructs the necessary unitary operators.
    It is based on the qubit tapering approach reported in https://arxiv.org/abs/1701.08213.

    Arguments:

    hamiltonian: QubitOperator
        The qubit Hamiltonian whose Z2 symmetries we want to find.

    taper_from_least: bool
        If True/False, the qubits to be tapered off will have the smallest/largest id numbers possible.

    debug: bool
        If True, the function returns the "generators" and "unitaries" lists described below, as well.

    Returns:

    generators: list of QubitOperator objects
        Qubit representation of set of independent generators of the symmetry group of the qubit Hamiltonian.

    sigma_x: list of integers
        The indices of the sigma_x Pauli gates that each generator will be transformed to.
        The indices coincide with the indices of the qubits to be tapered off.

    unitaries: list of QubitOperator objects
        The intermediate Clifford operators that transform each generator to a given sigma_x Pauli gate.

    unitary: QubitOperator
        The final Clifford operator that performs the desired similarity transformation.
    """

    n_qubits = hamiltonian.num_qubits()
    n_strings = len(hamiltonian.terms())

    prt_chck_mtrx = construct_parity_check_matrix(n_qubits, n_strings, hamiltonian)

    basis = find_parity_check_matrix_kernel(n_qubits, n_strings, prt_chck_mtrx)

    commute = find_commutation_matrix(basis)

    SGSO_basis, SGSO_commute = Symplectic_Gram_Schmidt_Orthogonalization(basis, commute)

    generators_binary = find_maximal_Abelian_subgroup(
        n_qubits, SGSO_basis, SGSO_commute, taper_from_least
    )

    # Translate binary vectors back to Pauli strings to obtain the generators of the symmetry group
    generators = []
    for i in range(generators_binary.shape[0]):
        pauli = qf.Circuit()
        for j in reversed(range(n_qubits)):
            if generators_binary[i, j] != 0 or generators_binary[i, j + n_qubits] != 0:
                if (
                    generators_binary[i, j] == 1
                    and generators_binary[i, j + n_qubits] == 1
                ):
                    gate_type = "Y"
                elif generators_binary[i, j] == 1:
                    gate_type = "X"
                elif generators_binary[i, j + n_qubits] == 1:
                    gate_type = "Z"
                pauli.add_gate(qf.gate(gate_type, n_qubits - j - 1))
        generators.append(pauli)

    # Find which sigma_x gate will be paired with which generator
    sigma_x = np.zeros((generators_binary.shape[0]), dtype=np.uint16)

    if taper_from_least:
        _range = range(n_qubits)
    else:
        _range = reversed(range(n_qubits))
    for i in _range:
        if np.argwhere(generators_binary[:, n_qubits + i]).shape[0] == 1:
            sigma_x[np.argwhere(generators_binary[:, n_qubits + i])] = n_qubits - i - 1

    # Reorder sigma_x operators in descending order of qubit identity. Do the same with the generators for consistency.

    reorder = np.argsort(sigma_x)
    sigma_x = sigma_x[reorder][::-1]
    generators = [generators[i] for i in reorder[::-1]]

    # Construct individual unitary matrices
    unitaries = []

    for idx, generator in enumerate(generators):
        circ = qf.Circuit()
        circ.add_gate(qf.gate("X", sigma_x[idx]))
        oprtr = qf.QubitOperator()
        oprtr.add_term(1 / np.sqrt(2), generator)
        oprtr.add_term(1 / np.sqrt(2), circ)
        unitaries.append(oprtr)

    # Construct final unitary matrix
    unitary = qf.QubitOperator()
    unitary.add_op(unitaries[0])
    for op in unitaries[1:]:
        unitary.operator_product(op, True, True)

    if debug:
        return generators, sigma_x, unitaries, unitary
    else:
        return sigma_x, unitary


def construct_parity_check_matrix(n_qubits, n_strings, hamiltonian):
    """
    This function constructs the parity check matrix corresponding to the given qubit Hamiltonian.
    The rows of the parity check matrix correspond to Pauli strings of the qubit Hamiltonian.
    The columns of the parity check matrix correspond to the Z_(n_qubits - 1), ..., Z_0, X_(n_qubits - 1), ..., X_0 Pauli gates.

    Arguments:

    n_qubits: int
        The number of qubits of the system.

    n_strings: int
        The number of Pauli strings contained in the qubit Hamiltonian

    hamiltonian: QubitOperator
        The qubit Hamiltonian whose Z2 symmetries we want to find.

    Returns:

    prt_chck_mtrx: uint8 numpy array
        The parity check matrix.
    """

    ## The rows of the parity check matrix correspond to Pauli strings of the qubit Hamiltonian.
    ## The columns of the parity check matrix correspond to the Z_(n_qubits - 1), ..., Z_0, X_(n_qubits - 1), ..., X_0 Pauli gates.
    prt_chck_mtrx = np.zeros((n_strings, 2 * n_qubits), dtype=np.uint8)
    for i, (_, string) in enumerate(hamiltonian.terms()):
        for pauli in string.gates():
            XYZ_gate = pauli.gate_id()
            trgt = pauli.target()
            if XYZ_gate in ["Y", "Z"]:
                prt_chck_mtrx[i, n_qubits - trgt - 1] = 1
            if XYZ_gate in ["X", "Y"]:
                prt_chck_mtrx[i, 2 * n_qubits - trgt - 1] = 1

    return prt_chck_mtrx


def find_parity_check_matrix_kernel(n_qubits, n_strings, prt_chck_mtrx):
    """
    This function finds a basis of the kernel of the parity check matrix (see
    https://en.wikipedia.org/wiki/Kernel_(linear_algebra)#Computation_by_Gaussian_elimination).

    Arguments:

    n_qubits: int
        The number of qubits of the system.

    n_strings: int
        The number of Pauli strings contained in the qubit Hamiltonian

    prt_chck_mtrx: uint8 numpy array
        The parity check matrix.

    Returns:

    basis: uint8 numpy array
        The rows of this matrix constitute a basis of the kernel of the parity check matrix.
    """

    ## Augment the (n_strings) × (2 * n_qubits) parity check matrix by the (2 * n_qubits) × (2 * n_qubits) identity matrix

    prt_chck_mtrx_aug = np.concatenate(
        (prt_chck_mtrx, np.identity(2 * n_qubits, dtype=np.uint8)), axis=0
    )

    ## Find column echelon form (CEF) of augmented matix.

    ### We only work with the columns of the parity check matrix that have not been transformed yet to CEF.
    ## This is controlled by the "idx" variable
    idx = 0

    for row in range(n_strings):
        ### Find a non-zero row
        nonzero_indices = np.argwhere(prt_chck_mtrx_aug[row, idx:])
        if nonzero_indices.shape[0] > 0:
            column_to_be_CEFed = nonzero_indices[0].item() + idx
            ### The first column that is non-zero in this row is the next column to "CEF" - swap positions accordingly
            prt_chck_mtrx_aug[:, [column_to_be_CEFed, idx]] = prt_chck_mtrx_aug[
                :, [idx, column_to_be_CEFed]
            ]
            ### Eliminate remaining non-zero row elements in the part of the parity check matrix that is not in CEF
            for i in range(1, nonzero_indices.shape[0]):
                column_not_in_CEF = nonzero_indices[i].item() + idx
                prt_chck_mtrx_aug[:, column_not_in_CEF] = np.bitwise_xor(
                    prt_chck_mtrx_aug[:, idx], prt_chck_mtrx_aug[:, column_not_in_CEF]
                )
            ### decrease the number of columns that need to be checked
            idx += 1

    ## Find basis vectors of kernel by checking for zero columns in the CEF of the parity check matrix

    basis = np.zeros((0, 2 * n_qubits), dtype=np.uint8)
    # The first idx columns are nonzero by construction
    for column in range(idx, 2 * n_qubits):
        row = n_strings
        if not np.any(prt_chck_mtrx_aug[:row, column]):
            new_basis_vector = prt_chck_mtrx_aug[row:, column].reshape(
                (1, 2 * n_qubits)
            )
            basis = np.concatenate((basis, new_basis_vector), axis=0)

    return basis


def find_commutation_matrix(basis):
    """
    This function constructs the "commute" binary matrix from a given set of binary basis vectors.

    Arguments:

    basis: uint8 numpy array
        The rows of this matrix constitute a basis of the kernel of the parity check matrix.

    Returns:

    commute: uint8 numpy array
        commute[i,j] = 0 (1) means that the ith and jth basis vectors commute (anticommute).
    """

    ## Construct symplectic bilinear form matrix B, which acts on binary vectors.
    ## v * B * w.T = 0/1 means v and w commute/anticommute.

    zero_matrix = np.zeros(
        (int(basis.shape[1] / 2), int(basis.shape[1] / 2)), dtype=np.uint8
    )
    identity_matrix = np.identity(int(basis.shape[1] / 2), dtype=np.uint8)
    blnr_frm_mtrx = np.block(
        [[zero_matrix, identity_matrix], [identity_matrix, zero_matrix]]
    )

    ## For all pairs of ker(E) basis vectors, compute commute[v, w] = v * B * w.T

    commute = np.matmul(basis, np.matmul(blnr_frm_mtrx, basis.T))
    commute %= 2

    return commute


def Symplectic_Gram_Schmidt_Orthogonalization(basis, commute):
    """
    This function performs a symplectic Gram-Schmidt orthogonalization over GF(2) on a given
    set of binary basis vectors (see https://doi.org/10.1103/PhysRevA.79.062322).

    Arguments:

    basis: uint8 numpy array
        The rows of this matrix constitute a basis of the kernel of the parity check matrix.

    commute: uint8 numpy array
        commute[i,j] = 0 (1) means that the ith and jth basis vectors commute (anticommute).

    Returns:

    SGSO_basis: uint8 numpy array
        The rows of this matrix constitute the Gram-Schmidt orthogonalized basis of the
        kernel of the parity check matrix.

    SGSO_commute: uint8 numpy array
        SGSO_commute[i,j] = 0 (1) means that the ith and jth Gram-Schmidt orthogonalized basis
        vectors commute (anticommute).

    """

    SGSO_basis = basis.copy()
    SGSO_commute = commute.copy()

    # List of indices of basis vectors that have already been Gram-Schmidt orthogonalized
    processed = set()

    for pauli_1 in range(SGSO_basis.shape[0]):
        if pauli_1 not in processed:
            processed.add(pauli_1)
            anticommute = np.argwhere(SGSO_commute[pauli_1, :])
            if anticommute.shape[0] != 0:
                for i in anticommute:
                    pauli_2 = int(i)
                    if pauli_2 not in processed:
                        processed.add(pauli_2)
                        for pauli in range(SGSO_basis.shape[0]):
                            if pauli not in processed:
                                if SGSO_commute[pauli, pauli_2]:
                                    SGSO_basis[pauli] ^= SGSO_basis[pauli_1]
                                if SGSO_commute[pauli, pauli_1]:
                                    SGSO_basis[pauli] ^= SGSO_basis[pauli_2]
                        break

                SGSO_commute = find_commutation_matrix(SGSO_basis)

    return SGSO_basis, SGSO_commute


def find_maximal_Abelian_subgroup(n_qubits, basis, commute, taper_from_least):
    """
    This function finds the maximal Abelian subgroup of the kernel of the parity check matrix.
    The generators of the maximal Abelian subgroup are the generators of the symmetry group
    of the qiven qubit Hamiltonian.

    Arguments:

    n_qubits: int
        The number of qubits of the system.

    basis: uint8 numpy array
        The rows of this matrix constitute a basis of the kernel of the parity check matrix.

    commute: uint8 numpy array
        commute[i,j] = 0 (1) means that the ith and jth basis vectors commute (anticommute).

    taper_from_least: bool
        If True/False, the qubits to be tapered off will have the smallest/largest id numbers possible.

    Returns:

    generators_binary: uint8 numpy array
        The rows of this matrix are the generators, in binary representation, of the symmetry group of the qubit Hamiltonian.
    """

    ## Find the maximal abelian subgroup iteratively. This is done as follows:
    ## 1) Find non-zero row of commute with smallest Hamming weight.
    ##    The corresponding basis vector e commutes with most other basis vectors.
    ## 2) Find basis vectors that do not commute with e and remove them.
    ## 3) Reconstruct commute matrix with remaining basis vectors.
    ## 4) If commute == 0 exit otherwise go to step 1) and repeat.

    basis_tmp = basis

    while not np.all(commute == 0):
        ### Maximum possible Hamming weight of a given binary vector
        hmng = 2 * n_qubits

        idx = 0
        for row in range(commute.shape[0]):
            ### Find the basis vector e that commutes with most of the remaining basis vectors
            hmng_tmp = np.sum(commute[row, :])
            if hmng_tmp > 0 and hmng > hmng_tmp:
                hmng = hmng_tmp
                idx = row

        ### Identify basis vectors that do not commute with e and eliminate them.
        nonzero_indices = np.argwhere(commute[idx, :])
        basis_tmp = np.delete(basis_tmp, nonzero_indices, axis=0)

        ### Construct commute matrix for the next iteration.

        commute = find_commutation_matrix(basis_tmp)

    generators_binary = basis_tmp

    ## Multiply the generators among themselves to create a new generating set that is
    ## compatible with tapering off the qubits with smallest/largest indices possible.
    ## Compatible here means "satisfy the (anti)commutation relations".

    idx = 0
    lst = []
    if taper_from_least:
        _range = reversed(range(n_qubits, 2 * n_qubits))
    else:
        _range = range(n_qubits, 2 * n_qubits)
    for column in _range:
        nonzero_indices = np.argwhere(generators_binary[:, column])
        if nonzero_indices.shape[0] == 1:
            if nonzero_indices[0] not in lst:
                lst.append(nonzero_indices[0].item())
                idx += 1
        else:
            for i in range(nonzero_indices.shape[0]):
                if nonzero_indices[i] not in lst:
                    lst.append(int(nonzero_indices[i]))
                    for el in filter(
                        lambda el: el not in nonzero_indices[i], nonzero_indices
                    ):
                        generators_binary[int(el), :] = np.bitwise_xor(
                            generators_binary[int(el), :],
                            generators_binary[int(nonzero_indices[i]), :],
                        )
                    idx += 1
                    break
        if idx == generators_binary.shape[0]:
            break

    return generators_binary


def taper_operator(tapered_qubits, sign, operator, unitary):
    """
    This function uses the unitary matrix obtained using find_Z2_symmetries to transform a given operator.
    After the unitary transformation, the resulting operator acts on the qubits to be tapered off by at most
    a single sigma_x operator. The sigma_x gates acting on qubits to be tapered off are replaced by their
    +/- 1 eigenvalues, which are provided by the user.

    Arguments:

    tapered_qubits: list of integers
        The indices of the qubits to be tapered off.

    sign: list of +/- 1 of length len(tapered_qubits).
        Specifies the symmetry subspace of the Hamiltonian that we are interested in.
        For each qubit to be tapered off, we can either be in the +1 or -1 subspace.

    operator: QubitOperator
        The operator that we wish to transform. For example, the qubit Hamiltonian or the UCC wave operator.

    unitary: QubitOperator
        The unitary operator that will perform the dersired transformation.

    Returns:

    operator_tapered: QubitOperator
        The tapered operator.
    """

    # Validating 'sign' argument

    if not all(i == 1 or i == -1 for i in sign):
        raise ValueError("The signs should be either 1 or -1!")
    if not len(sign) == len(tapered_qubits):
        raise ValueError(
            "There should be as many signs as the number of tapered qubits!"
        )

    # Similarity transform the given operator using the fact that the unitary matrix is also Hermitian

    ## Create copies of these operators to prevent changing the original values
    unitary_tmp = qf.QubitOperator()
    unitary_tmp.add_op(unitary)
    operator_tmp = qf.QubitOperator()
    operator_tmp.add_op(operator)

    operator_tmp.operator_product(unitary_tmp, True, True)
    unitary_tmp.operator_product(operator_tmp, True, True)
    operator_tmp = unitary_tmp

    # Create a QForte QubitOperator object that will store the tapered operator
    operator_tapered = qf.QubitOperator()

    for i, (coeff, circuit) in enumerate(operator_tmp.terms()):
        tapered_circuit = qf.Circuit()
        # List that will store the coefficients modified according to the sign structure provided by the user
        for pauli in circuit.gates():
            if pauli.target() in tapered_qubits:
                coeff *= sign[np.argwhere(tapered_qubits == pauli.target()).item()]
            else:
                # Adjust gate targets to account for tapered qubits
                idx = np.count_nonzero(pauli.target() > tapered_qubits)
                target = pauli.target() - idx
                gate = pauli.gate_id()
                tapered_circuit.add_gate(qf.gate(gate, target))

        operator_tapered.add_term(coeff, tapered_circuit)

    operator_tapered.simplify(True)

    return operator_tapered


def taper_reference(tapered_qubits, ref):
    """
    This function removes the designated qubits from a given reference, ref.
    WARNING: The code assumes that the tapered_qubits list is in descending order.

    Arguments:

    tapered_qubits: list of integers
        The indices of the qubits to be tapered off.

    ref: list of integers
        A bit string representing the reference qubit state.

    Returns:

    ref_tapered: list of integers
        A bit string representing the tapered reference qubit state.
    """

    ref_tapered = deepcopy(ref)

    for i in tapered_qubits:
        del ref_tapered[i]

    return ref_tapered
