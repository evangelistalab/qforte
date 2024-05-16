#include <map>
#include <random>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <cmath>

#include "fmt/format.h"

#include "qubit_basis.h"
#include "circuit.h"
#include "gate.h"
#include "helpers.h"
#include "qubit_operator.h"
#include "qubit_op_pool.h"
#include "timer.h"
#include "sparse_tensor.h"

#include "computer.h"

#if defined(_OPENMP)
#include <omp.h>
extern const bool parallelism_enabled = true;
#else
extern const bool parallelism_enabled = false;
#endif

Computer::Computer(int nqubit, double print_threshold)
    : nqubit_(nqubit), print_threshold_(print_threshold) {
    nbasis_ = std::pow(2, nqubit_);
    basis_.assign(nbasis_, QubitBasis());
    coeff_.assign(nbasis_, 0.0);
    new_coeff_.assign(nbasis_, 0.0);
    for (size_t i = 0; i < nbasis_; i++) {
        basis_[i] = QubitBasis(i);
    }
    coeff_[0] = 1.;
}

std::complex<double> Computer::coeff(const QubitBasis& basis) { return coeff_[basis.index()]; }

void Computer::set_state(std::vector<std::pair<QubitBasis, double_c>> state) {
    std::fill(coeff_.begin(), coeff_.end(), 0.0);
    for (const auto& basis_c : state) {
        coeff_[basis_c.first.index()] = basis_c.second;
    }
}

void Computer::null_state() { std::fill(coeff_.begin(), coeff_.end(), 0.0); }

void Computer::reset() {
    null_state();
    coeff_[0] = 1.;
}

void Computer::apply_matrix(const std::vector<std::vector<std::complex<double>>>& Opmat) {
    // std::vector<std::complex<double>> old_coeff = coeff_;
    std::vector<std::complex<double>> result(nbasis_, 0.0);

    for (size_t I = 0; I < nbasis_; I++) {
        result[I] =
            std::inner_product(Opmat[I].begin(), Opmat[I].end(), coeff_.begin(),
                               std::complex<double>(0.0, 0.0), add_c<double>, complex_prod<double>);
    }
    coeff_ = result;
}

void Computer::apply_sparse_matrix(const SparseMatrix& Spmat) {
    // std::vector<std::complex<double>> old_coeff = coeff_;
    std::vector<std::complex<double>> result(nbasis_, 0.0);

    for (auto const& i : Spmat.to_vec_map()) {    // i-> [ size_t, SparseVector ]
        for (auto const& j : i.second.to_map()) { // j-> [size_t, complex double]
            result[i.first] += j.second * coeff_[j.first];
        }
    }
    coeff_ = result;
}

void Computer::apply_operator(const QubitOperator& qo) {
    std::vector<std::complex<double>> old_coeff = coeff_;
    std::vector<std::complex<double>> result(nbasis_, 0.0);
    for (const auto& term : qo.terms()) {
        apply_circuit(term.second);
        apply_constant(term.first);
        std::transform(result.begin(), result.end(), coeff_.begin(), result.begin(), add_c<double>);

        coeff_ = old_coeff;
    }
    coeff_ = result;
}

void Computer::apply_circuit(const Circuit& qc) {
    for (const auto& gate : qc.gates()) {
        apply_gate(gate);
    }
}

void Computer::apply_circuit_safe(const Circuit& qc) {
    for (const auto& gate : qc.gates()) {
        apply_gate_safe(gate);
    }
}

void Computer::apply_gate(const Gate& qg) {
    int nqubits = qg.nqubits();

    if (nqubits == 1) {
        apply_1qubit_gate(qg);
    }
    if (nqubits == 2) {
        apply_2qubit_gate(qg);
    }
}

void Computer::apply_gate_safe(const Gate& qg) {
    int nqubits = qg.nqubits();

    if (nqubits == 1) {
        apply_1qubit_gate_safe(qg);
    }
    if (nqubits == 2) {
        apply_2qubit_gate_safe(qg);
    }

    coeff_ = new_coeff_;
    std::fill(new_coeff_.begin(), new_coeff_.end(), 0.0);
}

void Computer::apply_constant(const std::complex<double> a) {
    std::transform(coeff_.begin(), coeff_.end(), coeff_.begin(),
                   std::bind(std::multiplies<std::complex<double>>(), std::placeholders::_1, a));
}

std::vector<double> Computer::measure_circuit(const Circuit& qc, size_t n_measurements) {
    // initialize a "QubitBasis_rotator" QC to represent the corresponding change
    // of basis
    Circuit QubitBasis_rotator;

    // copy old coefficients
    std::vector<std::complex<double>> old_coeff = coeff_;

    // TODO: make code more readable (Nick)
    // TODO: add gate lable not via enum? (Nick)
    // TODO: Acount for case where gate is only the identity

    for (const Gate& gate : qc.gates()) {
        size_t target_qubit = gate.target();
        std::string gate_id = gate.gate_id();
        if (gate_id == "Z") {
            Gate temp = make_gate("I", target_qubit, target_qubit);
            QubitBasis_rotator.add_gate(temp);
        } else if (gate_id == "X") {
            Gate temp = make_gate("H", target_qubit, target_qubit);
            QubitBasis_rotator.add_gate(temp);
        } else if (gate_id == "Y") {
            Gate temp = make_gate("Rx", target_qubit, target_qubit, M_PI / 2);
            QubitBasis_rotator.add_gate(temp);
        } else if (gate_id != "I") {
            // // // std::cout<<'unrecognized gate in operator!'<<std::endl;
        }
    }

    // apply QubitBasis_rotator circuit to 'trick' qcomputer into measureing in non Z basis
    apply_circuit(QubitBasis_rotator);
    std::vector<double> probs(nbasis_);
    for (size_t k = 0; k < nbasis_; k++) {
        probs[k] = std::real(std::conj(coeff_[k]) * coeff_[k]);
    }

    // random number device
    std::random_device rd;
    std::mt19937 gen(rd());

    // 'pick' an index from the discrete_distribution!
    std::discrete_distribution<> dd(std::begin(probs), std::end(probs));

    std::vector<double> results(n_measurements);

    for (size_t k = 0; k < n_measurements; k++) {
        size_t measurement = dd(gen);
        double value = 1.;
        for (const Gate& gate : qc.gates()) {
            size_t target_qubit = gate.target();
            value *= 1. - 2. * static_cast<double>(basis_[measurement].get_bit(target_qubit));
        }
        results[k] = value;
    }

    coeff_ = old_coeff;
    return results;
}

std::vector<std::vector<int>> Computer::measure_z_readouts_fast(size_t na, size_t nb,
                                                                size_t n_measurements) {

    std::vector<double> probs(nbasis_);
    for (size_t k = 0; k < nbasis_; k++) {
        probs[k] = std::real(std::conj(coeff_[k]) * coeff_[k]);
    }

    // random number device
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dd(std::begin(probs), std::end(probs));
    std::vector<std::vector<int>> readouts(n_measurements);

    for (size_t k = 0; k < n_measurements; k++) {
        size_t measurement = dd(gen);
        std::vector<int> temp_readout;
        for (size_t l = na; l < nb + 1; l++) {
            temp_readout.push_back(static_cast<int>(basis_[measurement].get_bit(l)));
        }
        readouts[k] = temp_readout;
    }
    return readouts;
}

std::vector<std::vector<int>> Computer::measure_readouts(const Circuit& qc, size_t n_measurements) {
    // initialize a "QubitBasis_rotator" QC to represent the corresponding change
    // of basis
    Circuit QubitBasis_rotator;

    // copy old coefficients
    std::vector<std::complex<double>> old_coeff = coeff_;

    for (const Gate& gate : qc.gates()) {
        size_t target_qubit = gate.target();
        std::string gate_id = gate.gate_id();
        if (gate_id == "Z") {
            Gate temp = make_gate("I", target_qubit, target_qubit);
            QubitBasis_rotator.add_gate(temp);
        } else if (gate_id == "X") {
            Gate temp = make_gate("H", target_qubit, target_qubit);
            QubitBasis_rotator.add_gate(temp);
        } else if (gate_id == "Y") {
            Gate temp = make_gate("Rx", target_qubit, target_qubit, M_PI / 2);
            QubitBasis_rotator.add_gate(temp);
        } else if (gate_id != "I") {
            // // // std::cout<<'unrecognized gate in operator!'<<std::endl;
        }
    }

    // apply QubitBasis_rotator circuit to 'trick' qcomputer into measuring in non Z basis
    apply_circuit(QubitBasis_rotator);
    std::vector<double> probs(nbasis_);
    for (size_t k = 0; k < nbasis_; k++) {
        probs[k] = std::real(std::conj(coeff_[k]) * coeff_[k]);
    }

    // random number device
    std::random_device rd;
    std::mt19937 gen(rd());

    // 'pick' an index from the discrete_distribution!
    std::discrete_distribution<> dd(std::begin(probs), std::end(probs));

    std::vector<std::vector<int>> readouts(n_measurements);

    for (size_t k = 0; k < n_measurements; k++) {
        size_t measurement = dd(gen);
        // double value = 1.;
        std::vector<int> temp_readout;
        for (const Gate& gate : qc.gates()) {
            size_t target_qubit = gate.target();
            temp_readout.push_back(static_cast<int>(basis_[measurement].get_bit(target_qubit)));
        }
        readouts[k] = temp_readout;
    }

    coeff_ = old_coeff;
    return readouts;
}

double Computer::perfect_measure_circuit(const Circuit& qc) {
    // initialize a "QubitBasis_rotator" QC to represent the corresponding change
    // of basis
    Circuit QubitBasis_rotator;

    // copy old coefficients
    std::vector<std::complex<double>> old_coeff = coeff_;

    for (const Gate& gate : qc.gates()) {
        size_t target_qubit = gate.target();
        std::string gate_id = gate.gate_id();
        if (gate_id == "Z") {
            Gate temp = make_gate("I", target_qubit, target_qubit);
            QubitBasis_rotator.add_gate(temp);
        } else if (gate_id == "X") {
            Gate temp = make_gate("H", target_qubit, target_qubit);
            QubitBasis_rotator.add_gate(temp);
        } else if (gate_id == "Y") {
            Gate temp = make_gate("Rx", target_qubit, target_qubit, M_PI / 2);
            QubitBasis_rotator.add_gate(temp);
        } else if (gate_id != "I") {
            // std::cout<<'unrecognized gate in operator!'<<std::endl;
        }
    }

    // apply QubitBasis_rotator circuit to 'trick' qcomputer into measureing in non Z basis
    apply_circuit(QubitBasis_rotator);

    double sum = 0.0;
    for (size_t k = 0; k < nbasis_; k++) {
        double value = 1.0;
        for (const Gate& gate : qc.gates()) {
            size_t target_qubit = gate.target();
            value *= 1. - 2. * static_cast<double>(basis_[k].get_bit(target_qubit));
        }
        sum += std::real(value * coeff(basis_[k]) * std::conj(coeff(basis_[k])));
    }

    coeff_ = old_coeff;
    return sum;
}

#include <iostream>

void Computer::apply_1qubit_gate_safe(const Gate& qg) {
    size_t target = qg.target();
    const auto& mat = qg.matrix();

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            auto op_i_j = mat[i][j];
            if (std::abs(op_i_j) > compute_threshold_) {
                for (const QubitBasis& basis_J : basis_) {
                    if (basis_J.get_bit(target) == j) {
                        QubitBasis basis_I = basis_J;
                        basis_I.set_bit(target, i);
                        new_coeff_[basis_I.index()] += op_i_j * coeff_[basis_J.index()];
                    }
                }
            }
        }
    }
    none_ops_++;
}

void Computer::apply_1qubit_gate(const Gate& qg) {
    size_t target = qg.target();
    const auto& mat = qg.matrix();

    const size_t block_size = std::pow(2, target);
    const size_t block_offset = 2 * block_size;

    // bit target goes from j -> i
    const auto op_0_0 = mat[0][0];
    const auto op_0_1 = mat[0][1];
    const auto op_1_0 = mat[1][0];
    const auto op_1_1 = mat[1][1];

    if ((std::abs(op_0_0) + std::abs(op_1_1) > compute_threshold_) and
        (std::abs(op_0_1) + std::abs(op_1_0) > compute_threshold_)) {
        // Case I: this matrix has diagonal and off-diagonal elements. Apply standard algorithm
        size_t block_start_0 = 0;
        size_t block_start_1 = block_size;
        size_t block_end_0 = block_start_0 + block_size;
        for (; block_end_0 <= nbasis_;) {
            for (size_t I0 = block_start_0, I1 = block_start_1; I0 < block_end_0; ++I0, ++I1) {
                const auto x0 = coeff_[I0];
                const auto x1 = coeff_[I1];
                coeff_[I0] = op_0_0 * x0 + op_0_1 * x1;
                coeff_[I1] = op_1_0 * x0 + op_1_1 * x1;
            }
            block_start_0 += block_offset;
            block_start_1 += block_offset;
            block_end_0 += block_offset;
        }
    } else if (std::abs(op_0_0) + std::abs(op_1_1) > compute_threshold_) {
        // Case II: this matrix has no off-diagonal elements. Apply optimized algorithm
        if (op_0_0 != 1.0) {
            // Case II-A: changes portion of coeff_ only if g_00 is not 1.0
            size_t block_start_0 = 0;
            size_t block_end_0 = block_start_0 + block_size;
            for (; block_end_0 <= nbasis_;) {
                for (size_t I0 = block_start_0; I0 < block_end_0; ++I0) {
                    coeff_[I0] = op_0_0 * coeff_[I0];
                }
                block_start_0 += block_offset;
                block_end_0 += block_offset;
            }
        }
        if (op_1_1 != 1.0) {
            // Case II-B: changes portion of coeff_ only if g_11 is not 1.0
            size_t block_start_1 = block_size;
            size_t block_end_1 = block_start_1 + block_size;
            for (; block_end_1 <= nbasis_;) {
                for (size_t I1 = block_start_1; I1 < block_end_1; ++I1) {
                    coeff_[I1] = op_1_1 * coeff_[I1];
                }
                block_start_1 += block_offset;
                block_end_1 += block_offset;
            }
        }
    } else {
        // Case III: this matrix has only off-diagonal elements.
        if (op_0_1 == op_1_0 == 1.0) {
            // Case III-A: Apply optimized algorithm for X gate
            size_t block_start_0 = 0;
            size_t block_end_0 = block_start_0 + block_size;
            for (; block_end_0 <= nbasis_;) {
                for (size_t I0 = block_start_0; I0 < block_end_0; ++I0) {
                    std::swap(coeff_[I0], coeff_[I0 + block_size]);
                }
                block_start_0 += block_offset;
                block_end_0 += block_offset;
            }
        } else {
            // Case III-B: this matrix has only off-diagonal elements. Apply optimized algorithm
            size_t block_start_0 = 0;
            size_t block_end_0 = block_start_0 + block_size;
            for (; block_end_0 <= nbasis_;) {
                for (size_t I0 = block_start_0; I0 < block_end_0; ++I0) {
                    const auto x0 = coeff_[I0];
                    coeff_[I0] = op_0_1 * coeff_[I0 + block_size];
                    coeff_[I0 + block_size] = op_1_0 * x0;
                }
                block_start_0 += block_offset;
                block_end_0 += block_offset;
            }
        }
    }
    none_ops_++;
}

void Computer::apply_2qubit_gate_safe(const Gate& qg) {
    const auto& two_qubits_basis = Gate::two_qubits_basis();

    size_t target = qg.target();
    size_t control = qg.control();
    const auto& mat = qg.matrix();

    for (size_t i = 0; i < 4; i++) {
        const auto i_c = two_qubits_basis[i].first;
        const auto i_t = two_qubits_basis[i].second;
        for (size_t j = 0; j < 4; j++) {
            const auto j_c = two_qubits_basis[j].first;
            const auto j_t = two_qubits_basis[j].second;
            auto op_i_j = mat[i][j];
            if (std::abs(op_i_j) > compute_threshold_) {
                // if (auto op_i_j = gate[i][j]; std::abs(op_i_j) > compute_threshold_) { // C++17
                for (const QubitBasis& basis_J : basis_) {
                    if ((basis_J.get_bit(control) == j_c) and (basis_J.get_bit(target) == j_t)) {
                        QubitBasis basis_I = basis_J;
                        basis_I.set_bit(control, i_c);
                        basis_I.set_bit(target, i_t);
                        new_coeff_[basis_I.index()] += op_i_j * coeff_[basis_J.index()];
                    }
                }
            }
        }
    }

    ntwo_ops_++;
}

void Computer::apply_2qubit_gate(const Gate& qg) {
    const size_t target = qg.target();
    const size_t control = qg.control();
    const auto& mat = qg.matrix();

    // bit target goes from j -> i
    const auto op_2_2 = mat[2][2];
    const auto op_2_3 = mat[2][3];
    const auto op_3_2 = mat[3][2];
    const auto op_3_3 = mat[3][3];

    if ((std::abs(mat[0][1]) + std::abs(mat[1][0]) < compute_threshold_) and (mat[0][0] == 1.0) and
        (mat[1][1] == 1.0)) {
        // Case 1: 2qubit gate is a control gate
        if (target < control) {
            // Case I-A: target bit index is smaller than control bit index
            const size_t outer_block_size = std::pow(2, control);
            const size_t outer_block_offset = 2 * outer_block_size;
            const size_t block_size = std::pow(2, target);
            const size_t block_offset = 2 * block_size;

            if ((std::abs(op_2_2) + std::abs(op_3_3) > compute_threshold_) and
                (std::abs(op_2_3) + std::abs(op_3_2) > compute_threshold_)) {
                // Case I: this matrix has diagonal and off-diagonal elements. Apply standard
                // algorithm
                size_t outer_block_end = outer_block_offset;
                size_t block_start_0 = outer_block_size;
                size_t block_start_1 = outer_block_size + block_size;
                size_t block_end_0 = outer_block_size + block_size;

                for (; outer_block_end <= nbasis_;) {
                    for (; block_end_0 <= outer_block_end;) {
                        for (size_t I0 = block_start_0, I1 = block_start_1; I0 < block_end_0;
                             ++I0, ++I1) {
                            const auto x0 = coeff_[I0];
                            const auto x1 = coeff_[I1];
                            coeff_[I0] = op_2_2 * x0 + op_2_3 * x1;
                            coeff_[I1] = op_3_2 * x0 + op_3_3 * x1;
                        }
                        block_start_0 += block_offset;
                        block_start_1 += block_offset;
                        block_end_0 += block_offset;
                    }
                    block_start_0 += outer_block_size;
                    block_start_1 += outer_block_size;
                    block_end_0 += outer_block_size;
                    outer_block_end += outer_block_offset;
                }
            } else if (std::abs(op_2_2) + std::abs(op_3_3) > compute_threshold_) {
                // Case II: this matrix has no off-diagonal elements. Apply optimized algorithm
                if (op_2_2 != 1.0) {
                    // Case II-A: changes portion of coeff_ only if g_00 is not 1.0
                    size_t outer_block_end = outer_block_offset;
                    size_t block_start_0 = outer_block_size;
                    size_t block_end_0 = outer_block_size + block_size;

                    for (; outer_block_end <= nbasis_;) {
                        for (; block_end_0 <= outer_block_end;) {
                            for (size_t I0 = block_start_0; I0 < block_end_0; ++I0) {
                                coeff_[I0] = op_2_2 * coeff_[I0];
                            }
                            block_start_0 += block_offset;
                            block_end_0 += block_offset;
                        }
                        block_start_0 += outer_block_size;
                        block_end_0 += outer_block_size;
                        outer_block_end += outer_block_offset;
                    }
                }
                if (op_3_3 != 1.0) {
                    // Case II-B: changes portion of coeff_ only if g_11 is not 1.0
                    size_t outer_block_end = outer_block_offset;
                    size_t block_start_1 = outer_block_size + block_size;
                    size_t block_end_1 = block_start_1 + block_size;

                    for (; outer_block_end <= nbasis_;) {
                        for (; block_end_1 <= outer_block_end;) {
                            for (size_t I1 = block_start_1; I1 < block_end_1; ++I1) {
                                coeff_[I1] = op_3_3 * coeff_[I1];
                            }
                            block_start_1 += block_offset;
                            block_end_1 += block_offset;
                        }
                        block_start_1 += outer_block_size;
                        block_end_1 += outer_block_size;
                        outer_block_end += outer_block_offset;
                    }
                }
            } else {
                // Case III: this matrix has only off-diagonal elements.
                if (op_2_3 == op_3_2 == 1.0) {
                    // Case III-A: Apply optimized algorithm for X gate
                    size_t outer_block_end = outer_block_offset;
                    size_t block_start_0 = outer_block_size;
                    size_t block_start_1 = outer_block_size + block_size;
                    size_t block_end_0 = outer_block_size + block_size;

                    for (; outer_block_end <= nbasis_;) {
                        for (; block_end_0 <= outer_block_end;) {
                            for (size_t I0 = block_start_0, I1 = block_start_1; I0 < block_end_0;
                                 ++I0, ++I1) {
                                std::swap(coeff_[I0], coeff_[I1]);
                            }
                            block_start_0 += block_offset;
                            block_start_1 += block_offset;
                            block_end_0 += block_offset;
                        }
                        block_start_0 += outer_block_size;
                        block_start_1 += outer_block_size;
                        block_end_0 += outer_block_size;
                        outer_block_end += outer_block_offset;
                    }
                } else {
                    // Case III-B: this matrix has only off-diagonal elements. Apply optimized
                    // algorithm
                    size_t outer_block_end = outer_block_offset;
                    size_t block_start_0 = outer_block_size;
                    size_t block_start_1 = outer_block_size + block_size;
                    size_t block_end_0 = outer_block_size + block_size;

                    for (; outer_block_end <= nbasis_;) {
                        for (; block_end_0 <= outer_block_end;) {
                            for (size_t I0 = block_start_0, I1 = block_start_1; I0 < block_end_0;
                                 ++I0, ++I1) {
                                const auto x0 = coeff_[I0];
                                coeff_[I0] = op_2_3 * coeff_[I1];
                                coeff_[I1] = op_3_2 * x0;
                            }
                            block_start_0 += block_offset;
                            block_start_1 += block_offset;
                            block_end_0 += block_offset;
                        }
                        block_start_0 += outer_block_size;
                        block_start_1 += outer_block_size;
                        block_end_0 += outer_block_size;
                        outer_block_end += outer_block_offset;
                    }
                }
            }
            ntwo_ops_++;
        } /* end if t < c */
        if (control < target) {
            // Case 1-B: control bit idx is smaller than target bit idx
            const size_t outer_block_size = 1 << target;
            const size_t outer_block_offset = outer_block_size << 1;
            const size_t block_size = 1 << control;
            const size_t block_offset = block_size << 1;

            if ((std::abs(op_2_2) + std::abs(op_3_3) > compute_threshold_) and
                (std::abs(op_2_3) + std::abs(op_3_2) > compute_threshold_)) {
                // Case I: this matrix has diagonal and off-diagonal elements. Apply standard
                // algorithm
                size_t outer_block_end = outer_block_offset;
                size_t block_start_0 = block_size;
                size_t block_start_1 = outer_block_size + block_size;
                size_t block_end_0 = block_offset;

                for (; outer_block_end <= nbasis_;) {
                    for (; block_end_0 <= outer_block_end - outer_block_size;) {
                        for (size_t I0 = block_start_0, I1 = block_start_1; I0 < block_end_0;
                             ++I0, ++I1) {
                            const auto x0 = coeff_[I0];
                            const auto x1 = coeff_[I1];
                            coeff_[I0] = op_2_2 * x0 + op_2_3 * x1;
                            coeff_[I1] = op_3_2 * x0 + op_3_3 * x1;
                        }
                        block_start_0 += block_offset;
                        block_start_1 += block_offset;
                        block_end_0 += block_offset;
                    }
                    block_start_0 += outer_block_size;
                    block_start_1 += outer_block_size;
                    block_end_0 += outer_block_size;
                    outer_block_end += outer_block_offset;
                }
            } else if (std::abs(op_2_2) + std::abs(op_3_3) > compute_threshold_) {
                // Case II: this matrix has no off-diagonal elements. Apply optimized algorithm
                if (op_2_2 != 1.0) {
                    // Case II-A: changes portion of coeff_ only if g_00 is not 1.0
                    for (int outer_block_start = 0; outer_block_start < nbasis_;
                         outer_block_start += outer_block_offset) {
                        for (int block_start = outer_block_start + block_size;
                             block_start < outer_block_start + outer_block_size;
                             block_start += block_offset) {
                            for (int I0 = block_start; I0 < block_start + block_size; ++I0) {
                                coeff_[I0] = op_2_2 * coeff_[I0];
                            }
                        }
                    }
                }
                if (op_3_3 != 1.0) {
                    // Case II-B: changes portion of coeff_ only if g_11 is not 1.0
                    size_t outer_block_end = outer_block_offset;
                    size_t block_start_1 = outer_block_size + block_size;
                    size_t block_end_1 = block_start_1 + block_size;

                    for (; outer_block_end <= nbasis_;) {
                        for (; block_end_1 <= outer_block_end;) {
                            for (size_t I1 = block_start_1; I1 < block_end_1; ++I1) {
                                coeff_[I1] = op_3_3 * coeff_[I1];
                            }
                            block_start_1 += block_offset;
                            block_end_1 += block_offset;
                        }
                        block_start_1 += outer_block_size;
                        block_end_1 += outer_block_size;
                        outer_block_end += outer_block_offset;
                    }
                }
            } else {
                // Case III: this matrix has only off-diagonal elements.
                if (op_2_3 == op_3_2 == 1.0) {
                    // Case III-A: Apply optimized algorithm for X gate
                    size_t outer_block_end_0 = outer_block_size;
                    size_t block_start_0 = block_size;
                    size_t block_start_1 = outer_block_size + block_size;
                    size_t block_end_0 = block_offset;

                    for (; outer_block_end_0 <= nbasis_;) {
                        for (; block_end_0 <= outer_block_end_0;) {
                            for (size_t I0 = block_start_0, I1 = block_start_1; I0 < block_end_0;
                                 ++I0, ++I1) {
                                std::swap(coeff_[I0], coeff_[I1]);
                            }
                            block_start_0 += block_offset;
                            block_start_1 += block_offset;
                            block_end_0 += block_offset;
                        }
                        block_start_0 += outer_block_size;
                        block_start_1 += outer_block_size;
                        block_end_0 += outer_block_size;
                        outer_block_end_0 += outer_block_offset;
                    }
                } else {
                    // Case III-B: this matrix has only off-diagonal elements. Apply optimized
                    // algorithm
                    size_t outer_block_end_0 = outer_block_size;
                    size_t block_start_0 = block_size;
                    size_t block_start_1 = outer_block_size + block_size;
                    size_t block_end_0 = block_offset;

                    for (; outer_block_end_0 <= nbasis_;) {
                        for (; block_end_0 <= outer_block_end_0;) {
                            for (size_t I0 = block_start_0, I1 = block_start_1; I0 < block_end_0;
                                 ++I0, ++I1) {
                                const auto x0 = coeff_[I0];
                                coeff_[I0] = op_2_3 * coeff_[I1];
                                coeff_[I1] = op_3_2 * x0;
                            }
                            block_start_0 += block_offset;
                            block_start_1 += block_offset;
                            block_end_0 += block_offset;
                        }
                        block_start_0 += outer_block_size;
                        block_start_1 += outer_block_size;
                        block_end_0 += outer_block_size;
                        outer_block_end_0 += outer_block_offset;
                    }
                }
            }
            ntwo_ops_++;
        } // end if c < t
    }     // end if controlled unitary
    else {
        // Case 2: 2qubit gate is a not a control gate, use standard algorithm
        const auto& two_qubits_basis = Gate::two_qubits_basis();

        for (size_t i = 0; i < 4; i++) {
            const auto i_c = two_qubits_basis[i].first;
            const auto i_t = two_qubits_basis[i].second;
            for (size_t j = 0; j < 4; j++) {
                const auto j_c = two_qubits_basis[j].first;
                const auto j_t = two_qubits_basis[j].second;
                auto op_i_j = mat[i][j];
                if (std::abs(op_i_j) > compute_threshold_) {
                    // if (auto op_i_j = gate[i][j]; std::abs(op_i_j) > compute_threshold_) { //
                    // C++17
                    for (const QubitBasis& basis_J : basis_) {
                        if ((basis_J.get_bit(control) == j_c) and
                            (basis_J.get_bit(target) == j_t)) {
                            QubitBasis basis_I = basis_J;
                            basis_I.set_bit(control, i_c);
                            basis_I.set_bit(target, i_t);
                            new_coeff_[basis_I.index()] += op_i_j * coeff_[basis_J.index()];
                        }
                    }
                }
            }
        }
        ntwo_ops_++;
        coeff_ = new_coeff_;
        std::fill(new_coeff_.begin(), new_coeff_.end(), 0.0);
    }
}

std::complex<double> Computer::direct_op_exp_val(const QubitOperator& qo) {
    // local_timer t;
    std::complex<double> result = 0.0;
    if (parallelism_enabled) {
        for (const auto& term : qo.terms()) {
            result += term.first * direct_pauli_circ_exp_val(term.second);
        }
    } else {
        std::vector<std::complex<double>> old_coeff = coeff_;
        apply_operator(qo);
        result =
            std::inner_product(old_coeff.begin(), old_coeff.end(), coeff_.begin(),
                               std::complex<double>(0.0, 0.0), add_c<double>, complex_prod<double>);

        coeff_ = old_coeff;
    }
    // timings_.push_back(std::make_pair("direct_op_exp_val", t.get()));
    return result;
}

std::vector<std::complex<double>> Computer::direct_oppl_exp_val(const QubitOpPool& qopl) {

    std::vector<std::complex<double>> results;

    for (const auto& pl_term : qopl.terms()) {
        results.push_back(direct_op_exp_val(pl_term.second) * pl_term.first);
    }

    return results;
}

std::vector<std::complex<double>> Computer::direct_idxd_oppl_exp_val(const QubitOpPool& qopl,
                                                                     const std::vector<int>& idxs) {

    std::vector<std::complex<double>> results;
    if (parallelism_enabled) {
        for (const auto& idx : idxs) {
            std::complex<double> val = direct_op_exp_val(qopl.terms()[idx].second);
            results.push_back(val * qopl.terms()[idx].first);
        }
    } else {
        std::vector<std::complex<double>> old_coeff = coeff_;
        for (const auto& idx : idxs) {
            apply_operator(qopl.terms()[idx].second);
            std::complex<double> val = std::inner_product(
                old_coeff.begin(), old_coeff.end(), coeff_.begin(), std::complex<double>(0.0, 0.0),
                add_c<double>, complex_prod<double>);

            results.push_back(val * qopl.terms()[idx].first);
            coeff_ = old_coeff;
        }
    }
    return results;
}

std::vector<std::complex<double>>
Computer::direct_oppl_exp_val_w_mults(const QubitOpPool& qopl,
                                      const std::vector<std::complex<double>>& mults) {

    std::vector<std::complex<double>> results;
    if (parallelism_enabled) {
        for (const auto& pl_term : qopl.terms()) {
            std::complex<double> result = 0.0;
            for (int l = 0; l < pl_term.second.terms().size(); l++) {
                std::complex<double> val =
                    mults[l] * pl_term.first * pl_term.second.terms()[l].first;
                result += val * direct_pauli_circ_exp_val(pl_term.second.terms()[l].second);
            }
            results.push_back(result);
        }
    } else {
        for (const auto& pl_term : qopl.terms()) {
            std::complex<double> result = 0.0;
            for (int l = 0; l < pl_term.second.terms().size(); l++) {
                std::complex<double> val =
                    mults[l] * pl_term.first * pl_term.second.terms()[l].first;
                result += val * direct_circ_exp_val(pl_term.second.terms()[l].second);
            }
            results.push_back(result);
        }
    }
    return results;
}

std::complex<double> Computer::direct_circ_exp_val(const Circuit& qc) {
    std::vector<std::complex<double>> old_coeff = coeff_;
    std::complex<double> result = 0.0;

    apply_circuit(qc);
    result =
        std::inner_product(old_coeff.begin(), old_coeff.end(), coeff_.begin(),
                           std::complex<double>(0.0, 0.0), add_c<double>, complex_prod<double>);

    coeff_ = old_coeff;
    return result;
}

std::complex<double> Computer::direct_pauli_circ_exp_val(const Circuit& qc) {
    /* Efficiency optimization that explains the structure of this function:
     * Because our gates are all Pauli, the operator that represents our circuit is a direct
     * product of products of Pauli gates acting on individual qubits. These Pauli-products
     * may generate a coefficient and may flip a bit.
     * Now, let's say that the lowest qubit we have a non-trivial gate acting on is gate N+1.
     * Its image under the operator is the identity for the first N gates, and that direct
     * product for the remaining gates.
     * Let's divide our basis states into blocks of 2^N. Those are contiguous by construction,
     * and that's true for any choice of N.
     * Recall that gates are stored in ascending order, where gate 1 is incremented first, then
     * gate 2, and so forth. Therefore, the image of each block of 2^N gates is also a contiguous
     * state of basis states. This depends on both the fact that our operator acts trivially
     * on them, and the way we order our basis states.
     * Furthermore, these image states are orthogonal to all basis states except themselves.
     * That gives us a very convenient trick to compute the direct product: once you've computed
     * the image of the basis state leading the first block, find the basis state it's not
     * orthogonal to, and you know how to do the same for the remaining states in the block. Just
     * take the inner product, which is nice and easy because it's contiguous in memory. Do this for
     * all blocks, and you're done.
     */
    std::complex<double> result = 0.0;
    int min_qb_idx = nqubit_ - 1;
    std::vector<int> x_idxs;
    std::vector<int> y_idxs;
    std::vector<int> z_idxs;

    for (const auto& gate : qc.gates()) {
        if (gate.target() < min_qb_idx) {
            min_qb_idx = gate.target();
        }
        if (gate.gate_id() == "Z") {
            z_idxs.push_back(gate.target());
        } else if (gate.gate_id() == "X") {
            x_idxs.push_back(gate.target());
        } else if (gate.gate_id() == "Y") {
            y_idxs.push_back(gate.target());
        } else {
            throw std::runtime_error("Not a valid pauli gate!");
        }
    }

    int block_size = std::pow(2, min_qb_idx);
    int n_blocks = int(nbasis_ / block_size);

#pragma omp parallel for reduction(+ : result)
    for (size_t n = 0; n < n_blocks; n++) {
        size_t I1 = n * block_size;
        size_t I2 = I1 + block_size;

        auto pauli_perms = get_pauli_permuted_idx(I1, x_idxs, y_idxs, z_idxs);

        auto it1 = std::next(coeff_.begin(), I1);
        auto it2 = std::next(coeff_.begin(), I2);
        auto it3 = std::next(coeff_.begin(), pauli_perms.first);

        result +=
            pauli_perms.second * std::inner_product(it1, it2, it3, std::complex<double>(0.0, 0.0),
                                                    add_c<double>, complex_prod<double>);
    }
    return result;
}

std::pair<size_t, std::complex<double>>
Computer::get_pauli_permuted_idx(size_t I, const std::vector<int>& x_idxs,
                                 const std::vector<int>& y_idxs, const std::vector<int>& z_idxs) {

    QubitBasis basis_I(I);
    std::complex<double> val = 1.0;
    std::complex<double> onei(0.0, 1.0);

    for (const auto& xi : x_idxs) {
        basis_I.flip_bit(xi);
    }

    for (const auto& yi : y_idxs) {
        val *= onei * (1.0 - 2.0 * basis_I.get_bit(yi));
        basis_I.flip_bit(yi);
    }

    for (const auto& zi : z_idxs) {
        val *= (1.0 - 2.0 * basis_I.get_bit(zi));
    }

    return std::make_pair(basis_I.index(), val);
}

std::complex<double> Computer::direct_gate_exp_val(const Gate& qg) {
    std::vector<std::complex<double>> coeff_temp = coeff_;
    std::complex<double> result = 0.0;
    int nqubits = qg.nqubits();

    if (nqubits == 1) {
        apply_1qubit_gate(qg);
    }
    if (nqubits == 2) {
        apply_2qubit_gate(qg);
    }
    result =
        std::inner_product(coeff_temp.begin(), coeff_temp.end(), new_coeff_.begin(),
                           std::complex<double>(0.0, 0.0), add_c<double>, complex_prod<double>);

    std::fill(new_coeff_.begin(), new_coeff_.end(), 0.0);
    return result;
}

std::string Computer::str() const {
    std::vector<std::string> terms;
    terms.push_back("Computer(");
    for (size_t i = 0; i < nbasis_; i++) {
        if (std::abs(coeff_[i]) >= print_threshold_) {
            terms.push_back(to_string(coeff_[i]) + " " + basis_[i].str(nqubit_));
        }
    }
    terms.push_back(")");
    return join(terms, "\n");
}

bool operator==(const Computer& qc1, const Computer& qc2) {
    if (qc1.get_nqubit() != qc2.get_nqubit()) {
        return false;
    }
    return qc1.get_coeff_vec() == qc2.get_coeff_vec();
}

std::complex<double> dot(const Computer& qc1, const Computer& qc2) {
    return std::inner_product(
        qc1.get_coeff_vec().begin(), qc1.get_coeff_vec().end(), qc2.get_coeff_vec().begin(),
        std::complex<double>(0.0), std::plus<>(),
        [](std::complex<double> a, std::complex<double> b) { return std::conj(a) * b; });
}
