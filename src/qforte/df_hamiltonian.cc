#include <map>
#include <random>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <iterator>

#include "tensor.h"
#include "sq_operator.h"
#include "blas_math.h"

#include "df_hamiltonian.h"


DFHamiltonian::DFHamiltonian(
    int nel, 
    int norb, 
    Tensor& ff_eigenvalues,
    Tensor& one_body_squares,
    Tensor& one_body_ints,
    Tensor& one_body_correction,
    std::vector<Tensor>& scaled_density_density_matrices,
    std::vector<Tensor>& basis_change_matrices,
    std::vector<Tensor>& trotter_basis_change_matrices
    ) : 
        nel_(nel), 
        norb_(norb),
        ff_eigenvalues_(ff_eigenvalues),
        one_body_squares_(one_body_squares),
        one_body_ints_(one_body_ints),
        one_body_correction_(one_body_correction),
        scaled_density_density_matrices_(scaled_density_density_matrices),
        basis_change_matrices_(basis_change_matrices),
        trotter_basis_change_matrices_(trotter_basis_change_matrices){}

std::array<std::array<std::complex<double>, 2>, 2> DFHamiltonian::givens_matrix_elements(
        std::complex<double> a, 
        std::complex<double> b, 
        std::string which) 
{
        
        double cosine, sine;
        std::complex<double> phase(1.0, 0.0);

        if (std::abs(a) < 1.0e-11) {
            cosine = 1.0;
            sine = 0.0;
        } else if (std::abs(b) < 1.0e-11) {
            cosine = 0.0;
            sine = 1.0;
        } else {
            double denominator = std::sqrt(std::norm(a) + std::norm(b));
            cosine = std::abs(b) / denominator;
            sine = std::abs(a) / denominator;
            std::complex<double> sign_b = b / std::abs(b);
            std::complex<double> sign_a = a / std::abs(a);
            phase = sign_a * std::conj(sign_b);

            if (phase.imag() == 0) {
                phase = phase.real();
            }
        }

        std::array<std::array<std::complex<double>, 2>, 2> givens_rotation;

        if (which == "left") {
            if (std::abs(a.imag()) < 1.0e-11 && std::abs(b.imag()) < 1.0e-11) {
                givens_rotation = {{
                    {cosine, -phase * sine},
                    {phase * sine, cosine}
                }};
            } else {
                givens_rotation = {{
                    {cosine, -phase * sine},
                    {sine, phase * cosine}
                }};
            }
        } else if (which == "right") {
            if (std::abs(a.imag()) < 1.0e-11 && std::abs(b.imag()) < 1.0e-11) {
                givens_rotation = {{
                    {sine, phase * cosine},
                    {-phase * cosine, sine}
                }};
            } else {
                givens_rotation = {{
                    {sine, phase * cosine},
                    {cosine, -phase * sine}
                }};
            }
        } else {
            throw std::invalid_argument("\"which\" must be equal to \"left\" or \"right\".");
        }

        return givens_rotation;
}

// NOTE(Nick): note efficiet, speed up if this proves to be a bottleneck
void DFHamiltonian::givens_rotate(
    Tensor& op,
    const std::array<std::array<std::complex<double>, 2>, 2>& givens_rotation,
    size_t i, 
    size_t j, 
    std::string which) {

    op.square_error();
    Tensor op_new = op;
    size_t n = op.shape()[0];

    if (which == "row") {

        // Rotate rows i and j
        for (size_t k = 0; k < n; ++k) {
            size_t ik = n*i + k;
            size_t jk = n*j + k;
            op_new.data()[ik] = givens_rotation[0][0] * op.data()[ik] + givens_rotation[0][1] * op.data()[jk];
            op_new.data()[jk] = givens_rotation[1][0] * op.data()[ik] + givens_rotation[1][1] * op.data()[jk];
        }

    } else if (which == "col") {
        // Rotate columns i and j
        // NOTE(Nick): projably shuld just transpose and then do row wise for speed...
        for (size_t k = 0; k < n; ++k) {
            size_t ki = n*k + i;
            size_t kj = n*k + j;
            op_new.data()[ki] = givens_rotation[0][0] * op.data()[ki] + std::conj(givens_rotation[0][1]) * op.data()[kj];
            op_new.data()[kj] = givens_rotation[1][0] * op.data()[ki] + std::conj(givens_rotation[1][1]) * op.data()[kj];
        }

    } else {
        throw std::invalid_argument("\"which\" must be equal to \"row\" or \"col\".");
    }

    op = op_new;
}

std::tuple<
    std::vector<size_t>, 
    std::vector<size_t>, 
    std::vector<double>, 
    std::vector<double>, 
    std::vector<std::complex<double>>
> DFHamiltonian::givens_decomposition_square(
    const Tensor& unitary_matrix,
    const bool always_insert) {

    unitary_matrix.square_error();

    //deep copy I think?
    Tensor current_matrix = unitary_matrix; 
    int n = current_matrix.shape()[0];

    std::vector<size_t> i_vector;
    std::vector<size_t> j_vector;
    std::vector<double> theta_vector;
    std::vector<double> phi_vector;
    std::vector<std::complex<double>> diagonal(n);

    for (int k = 0; k < 2 * (n - 1) - 1; ++k) {
        int start_row, start_column;
        if (k < n - 1) {
            start_row = 0;
            start_column = n - 1 - k;
        } else {
            start_row = k - (n - 2);
            start_column = k - (n - 3);
        }

        std::vector<size_t> column_indices, row_indices;
        for (size_t col = start_column; col < n; col += 2) {
            column_indices.push_back(col);
        }
        for (size_t row = start_row; row < start_row + column_indices.size(); ++row) {
            row_indices.push_back(row);
        }

        for (size_t idx = 0; idx < row_indices.size(); ++idx) {
            size_t i = row_indices[idx];
            size_t j = column_indices[idx];

            // size_t ij_right = current_matrix.tidx_to_vidx({i, j});
            size_t ij_right = n*i + j;

            std::complex<double> right_element = std::conj(current_matrix.data()[ij_right]);
            // std::complex<double> right_element = std::conj(current_matrix[i][j]);

            if (always_insert || std::abs(right_element) > 1.0e-11) {

                // size_t ij_left = current_matrix.tidx_to_vidx({i, j-1});
                size_t ij_left = n * i + j - 1;
                std::complex<double> left_element = std::conj(current_matrix.data()[ij_left]);

                // std::complex<double> left_element = std::conj(current_matrix[i][j - 1]);

                auto givens_rotation = givens_matrix_elements(left_element, right_element, "right");

                double theta = std::asin(std::real(givens_rotation[1][0]));
                double phi = std::arg(givens_rotation[1][1]);
                
                i_vector.push_back(j - 1);
                j_vector.push_back(j);
                theta_vector.push_back(theta);
                phi_vector.push_back(phi);

                givens_rotate(current_matrix, givens_rotation, j - 1, j, "col");
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        diagonal[i] = current_matrix.data()[i*n + i];
    }

    return std::make_tuple(i_vector, j_vector, theta_vector, phi_vector, diagonal);
}

