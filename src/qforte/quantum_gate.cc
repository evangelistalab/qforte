#include "fmt/format.h"

#include "quantum_gate.h"

const std::vector<std::pair<size_t, size_t>> QuantumGate::two_qubits_basis_{
    {0, 0}, {0, 1}, {1, 0}, {1, 1}};
const std::vector<size_t> QuantumGate::index1{0, 1};
const std::vector<size_t> QuantumGate::index2{0, 1, 2, 3};

QuantumGate::QuantumGate(const std::string& label, size_t target, size_t control,
                         std::complex<double> gate[4][4])
    : label_(label), target_(target), control_(control) {
    for (const auto& i : index2) {
        for (const auto& j : index2) {
            gate_[i][j] = gate[i][j];
        }
    }
}

size_t QuantumGate::target() const { return target_; }

size_t QuantumGate::control() const { return control_; }

const complex_4_4_mat& QuantumGate::gate() const { return gate_; }

std::string QuantumGate::str() const {
    std::string s =
        fmt::format("{} gate, target qubit:{}, contol qubit:{}\n", label_, target_, control_);
    const std::vector<size_t>& index = (nqubits() == 1 ? index1 : index2);
    for (const auto& i : index2) {
        for (const auto& j : index2) {
            s += fmt::format("  {:+f} {:+f} i", std::real(gate_[i][j]), std::imag(gate_[i][j]));
        }
        s += '\n';
    }
    return s;
}

size_t QuantumGate::nqubits() const { return (target_ == control_ ? 1 : 2); }

const std::vector<std::pair<size_t, size_t>>& QuantumGate::two_qubits_basis() {
    return two_qubits_basis_;
}
