import numpy as np
from scipy.linalg import expm
import itertools
import qforte as qf

##                     ====> Helper Functions and such <====

# TODO(Nick): all this should be ported to the C++ side once hooks to LAPACK are implemented.
# I believe only an eigensolver is needed.
# This should all be handled in the constructor of the DFHamiltonain class. 

# The following is almost directly form openfirmion src/openfermion/circuits/low_rank.py
def low_rank_two_body_decomposition(
    two_body_coefficients, 
    truncation_threshold=1e-8, 
    final_rank=None, 
    spin_basis=True
):
    
    # Initialize N^2 by N^2 interaction array.
    one_body_correction, chemist_two_body_coefficients = get_chemist_two_body_coefficients(
        two_body_coefficients, 
        spin_basis
    )

    n_orbitals = chemist_two_body_coefficients.shape[0]
    full_rank = n_orbitals**2
    interaction_array = np.reshape(chemist_two_body_coefficients, (full_rank, full_rank))

    # Make sure interaction array is symmetric and real.
    asymmetry = np.sum(np.absolute(interaction_array - interaction_array.transpose()))
    imaginary_norm = np.sum(np.absolute(interaction_array.imag))
    if asymmetry > 1.0e-12 or imaginary_norm > 1.0e-12:
        raise TypeError('Invalid two-body coefficient tensor specification.')

    # Decompose with exact diagonalization.
    eigenvalues, eigenvectors = np.linalg.eigh(interaction_array)

    # Get one-body squares and compute weights.
    term_weights = np.zeros(full_rank)
    one_body_squares = np.zeros((full_rank, 2 * n_orbitals, 2 * n_orbitals), complex)

    # Reshape and add spin back in.
    for l in range(full_rank):
        one_body_squares[l] = np.kron(
            np.reshape(eigenvectors[:, l], (n_orbitals, n_orbitals)), np.eye(2)
        )
        term_weights[l] = abs(eigenvalues[l]) * np.sum(np.absolute(one_body_squares[l])) ** 2

    # Sort by weight.
    indices = np.argsort(term_weights)[::-1]
    eigenvalues = eigenvalues[indices]
    term_weights = term_weights[indices]
    one_body_squares = one_body_squares[indices]

    # Determine upper-bound on truncation errors that would occur.
    cumulative_error_sum = np.cumsum(term_weights)
    truncation_errors = cumulative_error_sum[-1] - cumulative_error_sum

    # Optionally truncate rank and return.
    if final_rank is None:
        max_rank = 1 + np.argmax(truncation_errors <= truncation_threshold)
    else:
        max_rank = final_rank
    truncation_value = truncation_errors[max_rank - 1]

    return (
        eigenvalues[:max_rank],
        one_body_squares[:max_rank],
        one_body_correction,
        truncation_value,
    )

# The following is almost directly form openfirmion src/openfermion/circuits/low_rank.py
def prepare_one_body_squared_evolution(one_body_matrix, spin_basis=False):

    # If the specification was in spin-orbitals, chop back down to spatial orbs
    # assuming a spin-symmetric interaction
    if spin_basis:
        n_modes = one_body_matrix.shape[0]
        alpha_indices = list(range(0, n_modes, 2))
        one_body_matrix = one_body_matrix[np.ix_(alpha_indices, alpha_indices)]

    # Diagonalize the one-body matrix.
    eigenvalues, eigenvectors = np.linalg.eigh(one_body_matrix)
        
    basis_transformation_matrix = np.conjugate(eigenvectors.transpose())

    # If the specification was in spin-orbitals, expand back
    if spin_basis:
        basis_transformation_matrix = np.kron(basis_transformation_matrix, np.eye(2))
        eigenvalues = np.kron(eigenvalues, np.ones(2))

    # Obtain the diagonal two-body matrix.
    density_density_matrix = np.outer(eigenvalues, eigenvalues)

    return density_density_matrix, basis_transformation_matrix


# The following is almost directly form openfirmion src/openfermion/circuits/low_rank.py
def get_chemist_two_body_coefficients(two_body_coefficients, spin_basis=True):
    # Initialize.
    n_orbitals = two_body_coefficients.shape[0]
    chemist_two_body_coefficients = np.transpose(two_body_coefficients, [0, 3, 1, 2])

    # If the specification was in spin-orbitals, chop down to spatial orbitals
    # assuming a spin-symmetric interaction.
    if spin_basis:
        n_orbitals = n_orbitals // 2
        alpha_indices = list(range(0, n_orbitals * 2, 2))
        beta_indices = list(range(1, n_orbitals * 2, 2))
        chemist_two_body_coefficients = chemist_two_body_coefficients[
            np.ix_(alpha_indices, alpha_indices, beta_indices, beta_indices)
        ]

    # Determine a one body correction in the spin basis from spatial basis.
    one_body_correction = np.zeros((2 * n_orbitals, 2 * n_orbitals), complex)
    for p, q, r, s in itertools.product(range(n_orbitals), repeat=4):
        for sigma, tau in itertools.product(range(2), repeat=2):
            if (q == r) and (sigma == tau):
                one_body_correction[2 * p + sigma, 2 * s + tau] -= chemist_two_body_coefficients[
                    p, q, r, s
                ]

    return one_body_correction, chemist_two_body_coefficients

# The following is almost directly form fqe OpenFermion-FQE/src/fqe/algorithm/low_rank.py
def first_factorization(
        tei,
        lmax,
        spin_basis,
        threshold
        ):
        r"""Factorize :math:`V = 1/2 \sum_{ijkl, st}V_{ijkl} is^ jt^ kt ls` by
        transforming to chemist notation.

        Args:
            threshold: threshold for factorization.

        Returns:
            Tuple of (eigenvalues of factors, one-body ops in factors, one
                      body correction).
        """

        # convert physics notation integrals into chemist notation
        # and determine the first low-rank factorization
        if spin_basis:
            (
                eigenvalues,
                one_body_squares,
                one_body_correction,
                _,
            ) = low_rank_two_body_decomposition(
                tei,
                truncation_threshold=threshold,
                final_rank=lmax,
                spin_basis=spin_basis)
            
        else:
            (
                eigenvalues,
                one_body_squares,
                one_body_correction,
                _,
            ) = low_rank_two_body_decomposition(
                0.5 * tei,  
                truncation_threshold=threshold,
                final_rank=lmax,
                spin_basis=spin_basis,
            )
        return eigenvalues, one_body_squares, one_body_correction

# The following is almost directly form fqe OpenFermion-FQE/src/fqe/algorithm/low_rank.py
def second_factorization(
        eigenvalues,
        one_body_squares,
        threshold=1.0e-12, # currently unused
):
    r"""
    Get Givens angles and DiagonalHamiltonian to simulate squared one-body.

    The goal here will be to prepare to simulate evolution under
    :math:`(\sum_{pq} h_{pq} a^{\dagger}_p a_q)^2` by decomposing as
    :math:`R e^{-i \sum_{pq} V_{pq} n_p n_q} R^\dagger` where
    :math:`R` is a basis transformation matrix.

    Args:
        eigenvalues: eigenvalues of 2nd quantized op

        one_body_squares: one-body-ops to square

    Returns:
        Tuple(List[np.ndarray], List[np.ndarray]) scaled-rho-rho spatial
        matrix and list of spatial basis transformations
    """
    
    scaled_density_density_matrices = []
    basis_change_matrices = []
    for j in range(len(eigenvalues)):
        
        (sdensity_density_matrix, sbasis_change_matrix) = prepare_one_body_squared_evolution(
            one_body_squares[j][::2, ::2], 
            spin_basis=False)
        
        scaled_density_density_matrices.append(np.real(eigenvalues[j]) * sdensity_density_matrix)
        basis_change_matrices.append(sbasis_change_matrix)

    return scaled_density_density_matrices, basis_change_matrices

# The following is almost directly form fqe OpenFermion-FQE/src/fqe/algorithm/low_rank.py
# if you want additional information or controll over thresholds in the second factorizaion
def get_l_and_m(
        one_body_squares, 
        second_factor_cutoff):
    """
    Determine the L rank and M rank for an integral matrix

    Returns:
        Return L and list of lists with M values for each L.
    """

    m_factors = []

    for l in range(one_body_squares.shape[0]):
        w, _ = np.linalg.eigh(one_body_squares[l][::2, ::2])
        # Determine upper-bound on truncation errors that would occur
        # if we dropped the eigenvalues lower than some cumulative error
        cumulative_error_sum = np.cumsum(sorted(np.abs(w))[::-1])
        truncation_errors = cumulative_error_sum[-1] - cumulative_error_sum
        max_rank = 1 + np.argmax(truncation_errors <= second_factor_cutoff)
        m_factors.append(max_rank)

    return one_body_squares.shape[0], m_factors

def time_scale_first_leaf(
        df_ham,
        evolution_time,
        ):

    g0_qf = df_ham.get_basis_change_matrices()[0]
    h1e_qf = df_ham.get_one_body_ints()
    h1e_cor_qf = df_ham.get_one_body_correction()

    g0_np = np.zeros(shape=g0_qf.shape(), dtype=np.complex128)
    h1e_np = np.zeros(shape=h1e_qf.shape(), dtype=np.complex128)
    h1e_cor_np = np.zeros(shape=h1e_cor_qf.shape(), dtype=np.complex128)

    for I in range(g0_qf.size()): g0_np.ravel()[I] = g0_qf.data()[I]
    for I in range(h1e_qf.size()): h1e_np.ravel()[I] = h1e_qf.data()[I]
    for I in range(h1e_cor_qf.size()): h1e_cor_np.ravel()[I] = h1e_cor_qf.data()[I]
    
    # og
    g0trot_np = g0_np @ expm(-1.0j * evolution_time * (h1e_np + h1e_cor_np))

    # g0trot_np = g0_np @ expm(-1.0j * evolution_time * (h1e_np)) #helps!

    print(h1e_qf)
    print(h1e_cor_qf)

    

    g0trot_qf = qf.Tensor(
        shape=np.shape(g0trot_np), 
        name='first_leaf_bais_change_trotter')
            
    g0trot_qf.fill_from_nparray(
        g0trot_np.ravel(), 
        np.shape(g0trot_np))
    

    # print(df_ham.get_trotter_basis_change_matrices()[0])

    df_ham.set_trotter_first_leaf_basis_chage(g0trot_qf)

    # print(df_ham.get_trotter_basis_change_matrices()[0])    

