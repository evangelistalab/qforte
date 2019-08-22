#include <map>
#include <random>
#include <algorithm>

#include "fmt/format.h"

#include "quantum_basis.h"
#include "quantum_circuit.h"
#include "quantum_gate.h"
#include "quantum_operator.h"

#include "quantum_computer.h"

QuantumComputer::QuantumComputer(int nqubit) : nqubit_(nqubit) {
    nbasis_ = std::pow(2, nqubit_);
    basis_.assign(nbasis_, QuantumBasis());
    coeff_.assign(nbasis_, 0.0);
    new_coeff_.assign(nbasis_, 0.0);
    for (size_t i = 0; i < nbasis_; i++) {
        basis_[i] = QuantumBasis(i);
    }
    coeff_[0] = 1.;
}

std::complex<double> QuantumComputer::coeff(const QuantumBasis& basis) {
    return coeff_[basis.add()];
}

void QuantumComputer::set_state(std::vector<std::pair<QuantumBasis, double_c>> state) {
    std::fill(coeff_.begin(), coeff_.end(), 0.0);
    for (const auto& basis_c : state) {
        coeff_[basis_c.first.add()] = basis_c.second;
    }
}

void QuantumComputer::zero_state() { std::fill(coeff_.begin(), coeff_.end(), 0.0); }

void QuantumComputer::apply_circuit(const QuantumCircuit& qc) {
    for (const auto& gate : qc.gates()) {
        apply_gate(gate);
    }
}

void QuantumComputer::apply_circuit_safe(const QuantumCircuit& qc) {
    for (const auto& gate : qc.gates()) {
        apply_gate_safe(gate);
    }
}

void QuantumComputer::apply_gate(const QuantumGate& qg) {
    int nqubits = qg.nqubits();

    if (nqubits == 1) {
        apply_1qubit_gate(qg);
    }
    if (nqubits == 2) {
        apply_2qubit_gate(qg);
    }
}

void QuantumComputer::apply_gate_safe(const QuantumGate& qg) {
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

std::vector<double> QuantumComputer::measure_circuit(const QuantumCircuit& qc,
                                                     size_t n_measurements) {
    // initialize a "Basis_rotator" QC to represent the corresponding change
    // of basis
    QuantumCircuit Basis_rotator;

    // copy old coefficients
    std::vector<std::complex<double>> old_coeff = coeff_;

    // TODO: make code more readable (Nick)
    // TODO: add gate lable not via enum? (Nick)
    // TODO: Acount for case where gate is only the identity

    for (const QuantumGate& gate : qc.gates()) {
        size_t target_qubit = gate.target();
        std::string gate_id = gate.gate_id();
        if (gate_id == "Z") {
            QuantumGate temp = make_gate("I", target_qubit, target_qubit);
            Basis_rotator.add_gate(temp);
        } else if (gate_id == "X") {
            QuantumGate temp = make_gate("H", target_qubit, target_qubit);
            Basis_rotator.add_gate(temp);
        } else if (gate_id == "Y") {
            QuantumGate temp = make_gate("Rzy", target_qubit, target_qubit);
            Basis_rotator.add_gate(temp);
        } else if (gate_id != "I") {
            // // // std::cout<<'unrecognized gate in operator!'<<std::endl;
        }
    }

    // apply Basis_rotator circuit to 'trick' qcomputer into measureing in non Z basis
    apply_circuit(Basis_rotator);
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
        for (const QuantumGate& gate : qc.gates()) {
            size_t target_qubit = gate.target();
            value *= 1. - 2. * static_cast<double>(basis_[measurement].get_bit(target_qubit));
        }
        results[k] = value;
    }

    coeff_ = old_coeff;
    return results;
}

double QuantumComputer::perfect_measure_circuit(const QuantumCircuit& qc) {
    // initialize a "Basis_rotator" QC to represent the corresponding change
    // of basis
    QuantumCircuit Basis_rotator;

    // copy old coefficients
    std::vector<std::complex<double>> old_coeff = coeff_;

    for (const QuantumGate& gate : qc.gates()) {
        size_t target_qubit = gate.target();
        std::string gate_id = gate.gate_id();
        if (gate_id == "Z") {
            QuantumGate temp = make_gate("I", target_qubit, target_qubit);
            Basis_rotator.add_gate(temp);
        } else if (gate_id == "X") {
            QuantumGate temp = make_gate("H", target_qubit, target_qubit);
            Basis_rotator.add_gate(temp);
        } else if (gate_id == "Y") {
            QuantumGate temp = make_gate("Rzy", target_qubit, target_qubit);
            Basis_rotator.add_gate(temp);
        } else if (gate_id != "I") {
            // std::cout<<'unrecognized gate in operator!'<<std::endl;
        }
    }

    // apply Basis_rotator circuit to 'trick' qcomputer into measureing in non Z basis
    apply_circuit(Basis_rotator);

    double sum = 0.0;
    for (size_t k = 0; k < nbasis_; k++){
        double value = 1.0;
        for (const QuantumGate& gate : qc.gates()) {
            size_t target_qubit = gate.target();
            value *= 1. - 2. * static_cast<double>(basis_[k].get_bit(target_qubit));
        }
        sum += std::real(value * coeff(basis_[k]) * std::conj(coeff(basis_[k])));
    }

    coeff_ = old_coeff;
    return sum;
}

#include <iostream>

void QuantumComputer::apply_1qubit_gate_safe(const QuantumGate& qg) {
    size_t target = qg.target();
    const auto& gate = qg.gate();

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            auto op_i_j = gate[i][j];
            if (std::abs(op_i_j) > compute_threshold_) {
                for (const QuantumBasis& basis_J : basis_) {
                    if (basis_J.get_bit(target) == j) {
                        QuantumBasis basis_I = basis_J;
                        basis_I.set_bit(target, i);
                        new_coeff_[basis_I.add()] += op_i_j * coeff_[basis_J.add()];
                    }
                }
            }
        }
    }
    none_ops_++;
}

void QuantumComputer::apply_1qubit_gate(const QuantumGate& qg) {
    size_t target = qg.target();
    const auto& gate = qg.gate();

    const size_t block_size = std::pow(2, target);
    const size_t block_offset = 2 * block_size;

    // bit target goes from j -> i
    const auto op_0_0 = gate[0][0];
    const auto op_0_1 = gate[0][1];
    const auto op_1_0 = gate[1][0];
    const auto op_1_1 = gate[1][1];

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

void QuantumComputer::apply_2qubit_gate_safe(const QuantumGate& qg) {
    const auto& two_qubits_basis = QuantumGate::two_qubits_basis();

    size_t target = qg.target();
    size_t control = qg.control();
    const auto& gate = qg.gate();

    for (size_t i = 0; i < 4; i++) {
        const auto i_c = two_qubits_basis[i].first;
        const auto i_t = two_qubits_basis[i].second;
        for (size_t j = 0; j < 4; j++) {
            const auto j_c = two_qubits_basis[j].first;
            const auto j_t = two_qubits_basis[j].second;
            auto op_i_j = gate[i][j];
            if (std::abs(op_i_j) > compute_threshold_) {
                // if (auto op_i_j = gate[i][j]; std::abs(op_i_j) > compute_threshold_) { // C++17
                for (const QuantumBasis& basis_J : basis_) {
                    if ((basis_J.get_bit(control) == j_c) and (basis_J.get_bit(target) == j_t)) {
                        QuantumBasis basis_I = basis_J;
                        basis_I.set_bit(control, i_c);
                        basis_I.set_bit(target, i_t);
                        new_coeff_[basis_I.add()] += op_i_j * coeff_[basis_J.add()];
                    }
                }
            }
        }
    }

    ntwo_ops_++;
}

void QuantumComputer::apply_2qubit_gate(const QuantumGate& qg) {
    const size_t target = qg.target();
    const size_t control = qg.control();
    const auto& gate = qg.gate();

    // bit target goes from j -> i
    const auto op_2_2 = gate[2][2];
    const auto op_2_3 = gate[2][3];
    const auto op_3_2 = gate[3][2];
    const auto op_3_3 = gate[3][3];

    if(( std::abs(gate[0][1]) + std::abs(gate[1][0]) < compute_threshold_ ) and
       ( gate[0][0] == 1.0 ) and ( gate[1][1] == 1.0 ) ) {
    // Case 1: 2qubit gate is a control gate
        if(target < control){
        // Case I-A: target bit index is smaller than control bit index
            const size_t outer_block_size = std::pow(2, control);
            const size_t outer_block_offset = 2 * outer_block_size;
            const size_t block_size = std::pow(2, target);
            const size_t block_offset = 2 * block_size;

            if ((std::abs(op_2_2) + std::abs(op_3_3) > compute_threshold_) and
                (std::abs(op_2_3) + std::abs(op_3_2) > compute_threshold_)) {
                // Case I: this matrix has diagonal and off-diagonal elements. Apply standard algorithm
                size_t outer_block_end = outer_block_offset;
                size_t block_start_0 = outer_block_size;
                size_t block_start_1 = outer_block_size + block_size;
                size_t block_end_0 = outer_block_size + block_size;

                for (; outer_block_end <= nbasis_;){
                    for (; block_end_0 <= outer_block_end;) {
                        for (size_t I0 = block_start_0, I1 = block_start_1; I0 < block_end_0; ++I0, ++I1) {
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

                    for(; outer_block_end <= nbasis_;){
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

                    for(; outer_block_end <= nbasis_;){
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

                    for (; outer_block_end <= nbasis_;){
                        for (; block_end_0 <= outer_block_end;) {
                            for (size_t I0 = block_start_0, I1 = block_start_1; I0 < block_end_0; ++I0, ++I1) {
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
                    // Case III-B: this matrix has only off-diagonal elements. Apply optimized algorithm
                    size_t outer_block_end = outer_block_offset;
                    size_t block_start_0 = outer_block_size;
                    size_t block_start_1 = outer_block_size + block_size;
                    size_t block_end_0 = outer_block_size + block_size;

                    for (; outer_block_end <= nbasis_;){
                        for (; block_end_0 <= outer_block_end;) {
                            for (size_t I0 = block_start_0, I1 = block_start_1; I0 < block_end_0; ++I0, ++I1) {
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
        }/* end if t < c */
        if(control < target) {
        // Case 1-B: control bit idx is smaller than target bit idx
            const size_t outer_block_size = std::pow(2, target);
            const size_t outer_block_offset = 2 * outer_block_size;
            const size_t block_size = std::pow(2, control);
            const size_t block_offset = 2 * block_size;

            if ((std::abs(op_2_2) + std::abs(op_3_3) > compute_threshold_) and
                (std::abs(op_2_3) + std::abs(op_3_2) > compute_threshold_)) {
                // Case I: this matrix has diagonal and off-diagonal elements. Apply standard algorithm
                size_t outer_block_end = outer_block_offset;
                size_t block_start_0 = block_size;
                size_t block_start_1 = outer_block_size + block_size;
                size_t block_end_0 = block_offset;

                for (; outer_block_end <= nbasis_;){
                    for (; block_end_0 <= outer_block_end-outer_block_size;) {
                        for (size_t I0 = block_start_0, I1 = block_start_1; I0 < block_end_0; ++I0, ++I1) {
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
                    size_t block_start_0 = block_size;
                    size_t block_start_1 = outer_block_size + block_size;
                    size_t block_end_0 = block_offset;

                    for(; outer_block_end <= nbasis_;){
                        for (; block_end_0 < outer_block_end;) {
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

                    for(; outer_block_end <= nbasis_;){
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

                    for (; outer_block_end_0 <= nbasis_;){
                        for (; block_end_0 <= outer_block_end_0;) {
                            for (size_t I0 = block_start_0, I1 = block_start_1; I0 < block_end_0; ++I0, ++I1) {
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
                    // Case III-B: this matrix has only off-diagonal elements. Apply optimized algorithm
                    size_t outer_block_end_0 = outer_block_size;
                    size_t block_start_0 = block_size;
                    size_t block_start_1 = outer_block_size + block_size;
                    size_t block_end_0 = block_offset;

                    for (; outer_block_end_0 <= nbasis_;){
                        for (; block_end_0 <= outer_block_end_0;) {
                            for (size_t I0 = block_start_0, I1 = block_start_1; I0 < block_end_0; ++I0, ++I1) {
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
    } // end if controlled unitary
    else{
    // Case 2: 2qubit gate is a not a control gate, use standard algorithm
        const auto& two_qubits_basis = QuantumGate::two_qubits_basis();

        for (size_t i = 0; i < 4; i++) {
            const auto i_c = two_qubits_basis[i].first;
            const auto i_t = two_qubits_basis[i].second;
            for (size_t j = 0; j < 4; j++) {
                const auto j_c = two_qubits_basis[j].first;
                const auto j_t = two_qubits_basis[j].second;
                auto op_i_j = gate[i][j];
                if (std::abs(op_i_j) > compute_threshold_) {
                    // if (auto op_i_j = gate[i][j]; std::abs(op_i_j) > compute_threshold_) { // C++17
                    for (const QuantumBasis& basis_J : basis_) {
                        if ((basis_J.get_bit(control) == j_c) and (basis_J.get_bit(target) == j_t)) {
                            QuantumBasis basis_I = basis_J;
                            basis_I.set_bit(control, i_c);
                            basis_I.set_bit(target, i_t);
                            new_coeff_[basis_I.add()] += op_i_j * coeff_[basis_J.add()];
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

std::complex<double> QuantumComputer::direct_op_exp_val(const QuantumOperator& qo) {
    std::complex<double> result = 0.0;
    for (const auto& term : qo.terms()) {
        result += term.first * direct_circ_exp_val(term.second);
    }
    return result;
}

std::complex<double> QuantumComputer::direct_circ_exp_val(const QuantumCircuit& qc) {
    std::vector<std::complex<double>> old_coeff = coeff_;
    std::complex<double> result = 0.0;

    apply_circuit(qc);
    result =
        std::inner_product(old_coeff.begin(), old_coeff.end(), coeff_.begin(),
                           std::complex<double>(0.0, 0.0), add_c<double>, complex_prod<double>);

    coeff_ = old_coeff;
    return result;
}

std::complex<double> QuantumComputer::direct_gate_exp_val(const QuantumGate& qg) {
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

std::vector<std::string> QuantumComputer::str() const {
    std::vector<std::string> terms;
    for (size_t i = 0; i < nbasis_; i++) {
        if (std::abs(coeff_[i]) >= print_threshold_) {
            terms.push_back(fmt::format("({:f} {:+f} i) {}", std::real(coeff_[i]),
                                        std::imag(coeff_[i]), basis_[i].str(nqubit_)));
        }
    }
    return terms;
}
