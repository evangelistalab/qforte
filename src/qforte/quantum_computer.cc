#include <map>
#include <random>
#include <algorithm>

#include "fmt/format.h"

#include "quantum_gate.h"
#include "quantum_computer.h"

std::string QuantumBasis::str(size_t nqubit) const {
    std::string s;
    s += "|";
    for (int i = 0; i < nqubit; ++i) {
        if (get_bit(i)) {
            s += "1";
        } else {
            s += "0";
        }
    }
    s += ">";
    return s;
}

void QuantumBasis::set(basis_t state) { state_ = state; }

QuantumBasis& QuantumBasis::insert(size_t pos) {
    basis_t temp(state_);
    state_ = state_ << 1;
    basis_t mask = (1 << pos) - 1;
    state_ = state_ ^ ((state_ ^ temp) & mask);
    return *this;
}

// std::vector<double> QuantumCircuit::get_parameters() {
//     // need a loop over only gates in state preparation circuit that
//     // have a parameter dependance (if gate_id == Rx, Ry, or Rz)
//     // TODO: make a indexing funciton using a map (Nick)
//     size_t param_idx = 0;
//     std::vector<double> params
//     for (auto& gate : gates_) {
//         std::string gate_id = gate.gate_id();
//         if (gate_id == "Rz") {
//
//             double param = gate.gate()[][];
//             gate = make_gate(gate_id, target_qubit, target_qubit, params[param_idx]);
//             param_idx++;
//         }
//     }
// }

void QuantumCircuit::set_parameters(const std::vector<double>& params) {
    // need a loop over only gates in state preparation circuit that
    // have a parameter dependance (if gate_id == Rx, Ry, or Rz)
    // TODO: make a indexing funciton using a map (Nick)
    size_t param_idx = 0;
    for (auto& gate : gates_) {
        std::string gate_id = gate.gate_id();
        if (gate_id == "Rz") {
            size_t target_qubit = gate.target();
            gate = make_gate(gate_id, target_qubit, target_qubit, params[param_idx]);
            param_idx++;
        }
    }
}

QuantumCircuit QuantumCircuit::adjoint() {
    QuantumCircuit qcirc_adjoint;
    for (auto& gate : gates_){
        qcirc_adjoint.add_gate(gate.adjoint())
    }
    std::reverse(std::begin(qcirc_adjoint.gates_), std::end(qcirc_adjoint.gates_));
    return qcirc_adjoint;
}

// void QuantumCircuit::set_reversed_gates() {
//     rev_copy_ = gates_;
//     std::reverse(std::begin(rev_copy_), std::end(rev_copy_));
// }

// std::vector<QuantumGate> reversed_gates() const {
//     std::vector<QuantumGate>rev_copy = gates_;
//     std::reverse(std::begin(rev_copy), std::end(rev_copy));
//     return rev_copy;
// }

std::vector<std::string> QuantumCircuit::str() const {
    std::vector<std::string> s;
    for (const auto& gate : gates_) {
        s.push_back(gate.str());
    }
    return s;
}

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

void QuantumComputer::zero_state() {
    std::fill(coeff_.begin(), coeff_.end(), 0.0);
}

void QuantumComputer::apply_circuit(const QuantumCircuit& qc) {
    for (const auto& gate : qc.gates()) {
        apply_gate(gate);
    }
}

void QuantumComputer::apply_gate(const QuantumGate& qg) {
    int nqubits = qg.nqubits();

    if (nqubits == 1) {
        apply_1qubit_gate(qg);
    }
    if (nqubits == 2){
        apply_2qubit_gate(qg);
    }

    coeff_ = new_coeff_;
    std::fill(new_coeff_.begin(), new_coeff_.end(), 0.0);
}

std::vector<double> QuantumComputer::measure_circuit(const QuantumCircuit& qc, size_t n_measurements) {
    // initialize a "Basis_rotator" QC to represent the corresponding change
    // of basis
    QuantumCircuit Basis_rotator;

    // copy old coefficients
    std::vector<std::complex<double>> old_coeff = coeff_;

    // TODO: make code more readable (Nick)
    // TODO: add gate lable not via enum? (Nick)
    // TODO: Acount for case where gate is only the identity

    for(const QuantumGate& gate : qc.gates()){
        size_t target_qubit = gate.target();
        std::string gate_id = gate.gate_id();
        if ( gate_id == "Z" ) {
            QuantumGate temp = make_gate("I", target_qubit, target_qubit);
            Basis_rotator.add_gate(temp);
        } else if ( gate_id == "X" ) {
            QuantumGate temp = make_gate("H", target_qubit, target_qubit);
            Basis_rotator.add_gate(temp);
        } else if (gate_id == "Y"){
            QuantumGate temp = make_gate("Rzy", target_qubit, target_qubit);
            Basis_rotator.add_gate(temp);
        } else if (gate_id != "I") {
            //std::cout<<'unrecognized gate in operator!'<<std::endl;
        }
    }

    // apply Basis_rotator circuit to 'trick' qcomputer into measureing in non Z basis
    apply_circuit(Basis_rotator);
    std::vector<double> probs(nbasis_);
    for (size_t k = 0; k < nbasis_; k++) { probs[k] = std::real(std::conj(coeff_[k]) * coeff_[k]); }

    // random number device
    std::random_device rd;
    std::mt19937 gen(rd());

    // 'pick' an index from the discrete_distribution!
    std::discrete_distribution<> dd(std::begin(probs), std::end(probs));

    std::vector<double> results(n_measurements);

    for(size_t k = 0; k < n_measurements; k++){
        size_t measurement = dd(gen);
        double value = 1.;
        for(const QuantumGate& gate : qc.gates()){
            size_t target_qubit = gate.target();
            value *= 1. - 2. * static_cast<double>(basis_[measurement].get_bit(target_qubit));
        }
        results[k] = value;
    }

    coeff_ = old_coeff;
    return results;
}

void QuantumComputer::apply_1qubit_gate(const QuantumGate& qg) {
    size_t target = qg.target();
    const auto& gate = qg.gate();

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            if (auto op_i_j = gate[i][j]; std::abs(op_i_j) > compute_threshold_) {
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
}

void QuantumComputer::apply_1qubit_gate_insertion(const QuantumGate& qg) {
    size_t target = qg.target();
    const auto& gate = qg.gate();

    QuantumBasis basis_I, basis_J, basis_K;
    size_t nbasis_minus1 = std::pow(2, nqubit_ - 1);
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            if (auto op_i_j = gate[i][j]; std::abs(op_i_j) > compute_threshold_) {
                for (size_t K = 0; K < nbasis_minus1; K++) {
                    basis_K.set(K);
                    basis_I = basis_J = basis_K.insert(target);
                    basis_I.set_bit(target, i);
                    basis_J.set_bit(target, j);
                    new_coeff_[basis_I.add()] += op_i_j * coeff_[basis_J.add()];
                }
            }
        }
    }
}

void QuantumComputer::apply_2qubit_gate(const QuantumGate& qg) {
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
            if (auto op_i_j = gate[i][j]; std::abs(op_i_j) > compute_threshold_) {
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
    result = std::inner_product(old_coeff.begin(), old_coeff.end(), coeff_.begin(),
                                std::complex<double>(0.0, 0.0), add_c<double>,complex_prod<double>
                                );

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
    if (nqubits == 2){
        apply_2qubit_gate(qg);
    }
    result = std::inner_product(coeff_temp.begin(), coeff_temp.end(), new_coeff_.begin(),
                                std::complex<double>(0.0, 0.0), add_c<double>,complex_prod<double>
                                );

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
