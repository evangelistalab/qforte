#ifndef _df_hamiltonain_h_
#define _df_hamiltonain_h_

#include <string>
#include <vector>

#include <array>
#include <complex>
#include <cmath>
#include <stdexcept>
#include <string>

#include "qforte-def.h" 
#include "tensor.h" 

class SQOperator;
class Tensor;

// using complex_2_2_mat = std::array<std::array<std::complex<double>, 2>, 2>;

class DFHamiltonian {
  public:

    // Constructor
    DFHamiltonian(
      int nel, 
      int norb, 
      Tensor& eigenvalues,
      Tensor& one_body_squares,
      Tensor& one_body_ints,
      Tensor& one_body_correction,
      std::vector<Tensor>& scaled_density_density_matrices,
      std::vector<Tensor>& basis_change_matrices,
      std::vector<Tensor>& trotter_basis_change_matrices
      );

    
    /* Compute the matrix elements of the Givens rotation that zeroes out one
    of two row entries.

    If `which='left'` then returns a matrix G such that

        G * [a  b]^T= [0  r]^T

    otherwise, returns a matrix G such that

        G * [a  b]^T= [r  0]^T

    where r is a complex number. 
    
    Args:
        a(complex or float): A complex number representing the upper row entry
        b(complex or float): A complex number representing the lower row entry
        which(string): Either 'left' or 'right', indicating whether to
            zero out the left element (first argument) or right element
            (second argument). Default is `left`.
    Returns:
        G(ndarray): A 2 x 2 numpy array representing the matrix G.
            The numbers in the first column of G are real. */

    static std::array<std::array<std::complex<double>, 2>, 2> givens_matrix_elements(
      std::complex<double> a, 
      std::complex<double> b,  
      std::string which = "left");

    // Apply a Givens rotation to coordinates i and j of an operator.
    static void givens_rotate(
        Tensor& op,
        const std::array<std::array<std::complex<double>, 2>, 2>& givens_rotation,
        size_t i, 
        size_t j, 
        std::string which = "row");


    /* Decompose a square matrix into a sequence of Givens rotations.

    The input is a square n x n matrix Q.
    Q can be decomposed as follows:

        Q = DU

    where U is unitary and D is diagonal.
    Furthermore, we can decompose U as

        U = G_k ... G_1

    where G_1, ..., G_k are complex Givens rotations.
    A Givens rotation is a rotation within the two-dimensional subspace
    spanned by two coordinate axes. Within the two relevant coordinate
    axes, a Givens rotation has the form

    $$
        \begin{pmatrix}
            \cos(\theta) & -e^{i \varphi} \sin(\theta) \\
            \sin(\theta) &     e^{i \varphi} \cos(\theta)
        \end{pmatrix}.
    $$

    decomposition (list[tuple]):
            A list of tuples of objects describing Givens
            rotations. The list looks like [(G_1, ), (G_2, G_3), ... ].
            The Givens rotations within a tuple can be implemented in parallel.
            The description of a Givens rotation is itself a tuple of the
            form $(i, j, \theta, \varphi)$, which represents a
            Givens rotation of coordinates
            $i$ and $j$ by angles $\theta$ and
            $\varphi$.

    diagonal (ndarray):
        A list of the nonzero entries of D */

    static std::tuple<
        std::vector<size_t>, // i vector
        std::vector<size_t>, // j vector
        std::vector<double>, // theta vector
        std::vector<double>, // phi vector
        std::vector<std::complex<double>> // vector that represents the diagonal
    > givens_decomposition_square(
        const Tensor& unitary_matrix,
        const bool always_insert = false);

    void set_trotter_first_leaf_basis_chage(Tensor& g0_trott) {
      trotter_basis_change_matrices_[0] = g0_trott;
    }

    size_t get_nel() const {return nel_;}

    size_t get_norb() const {return norb_;}

    /// Will just return copies for now...
    Tensor get_ff_eigenvalues() const {return ff_eigenvalues_;}

    Tensor get_one_body_squares() const {return one_body_squares_;}

    Tensor get_one_body_ints() const {return one_body_ints_;}

    Tensor get_one_body_correction() const {return one_body_correction_;}
    
    std::vector<Tensor> get_scaled_density_density_matrices() const {
      return scaled_density_density_matrices_;
    }

    std::vector<Tensor> get_basis_change_matrices() const {
      return basis_change_matrices_;
    }

    std::vector<Tensor> get_trotter_basis_change_matrices() const {
      return trotter_basis_change_matrices_;
    }
    
  private:
    /// the number of electrons
    size_t nel_;

    /// the number of spatial orbitals
    size_t norb_;

    /// egenvalues of the first factorization of the teis
    Tensor ff_eigenvalues_;

    /// the one body operator to squaure
    Tensor one_body_squares_;

    /// one body integrals, without any augmentation
    Tensor one_body_ints_;

    /// one body contribtions form the two body operator
    Tensor one_body_correction_;

    /// re-scaled density matricies, do NOT already account for time steps.
    std::vector<Tensor> scaled_density_density_matrices_;

    /// the matricies that basis changes
    /// the first element corresponds to the portion
    /// with the 1-body correction (the Zero leaf as Rob says)
    std::vector<Tensor> basis_change_matrices_;

    /// the matricies that define the givens rotations for trotterization
    /// the first element corresponds to the portion
    /// with the 1-body correction (the Zero leaf as Rob says)
    std::vector<Tensor> trotter_basis_change_matrices_;

    /// the threshold for doing operations with elements of gate matricies
    double compute_threshold_ = 1.0e-12;
};

#endif // _df_hamiltonian_h_
