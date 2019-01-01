#include <stdexcept>

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

QuantumGate make_gate(std::string type, size_t target, size_t control, double parameter) {
    using namespace std::complex_literals;
    if (target == control) {
        if (type == "X") {
            std::complex<double> gate[4][4]{
                {[1] = 1.0},
                {[0] = 1.0},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "Y") {
            std::complex<double> gate[4][4]{
                {[1] = -1.0i},
                {[0] = +1.0i},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "Z") {
            std::complex<double> gate[4][4]{
                {[0] = +1.0},
                {[1] = -1.0},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "H") {
            std::complex<double> c = 1.0 / std::sqrt(2.0);
            std::complex<double> gate[4][4]{
                {+c, +c},
                {+c, -c},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "R") {
            std::complex<double> c = std::exp(1.0i * parameter);
            std::complex<double> gate[4][4]{
                {1},
                {[1] = c},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "S") {
            std::complex<double> gate[4][4]{
                {1},
                {[1] = 1.0i},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "T") {
            std::complex<double> c = 1.0 / std::sqrt(2.0);
            std::complex<double> gate[4][4]{
                {1},
                {[1] = c * (1.0 + 1.0i)},
            };
            return QuantumGate(type, target, control, gate);
        }
    } else {
        if ((type == "cX") or (type == "CNOT")) {
            std::complex<double> gate[4][4]{
                {[0] = 1.0},
                {[1] = 1.0},
                {[3] = 1.0},
                {[2] = 1.0},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "cY") {
            std::complex<double> gate[4][4]{
                {[0] = 1.0},
                {[1] = 1.0},
                {[3] = -1.0i},
                {[2] = +1.0i},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "cZ") {
            std::complex<double> gate[4][4]{
                {[0] = 1.0},
                {[1] = 1.0},
                {[2] = 1.0},
                {[3] = -1.0},
            };
            return QuantumGate(type, target, control, gate);
        }
    }
    // TODO: throw an exception that propagates to Python
    std::string msg =
        fmt::format("make_quantum_gate()\ntype = {} is not a valid quantum gate type", type);
    throw std::invalid_argument(msg);
    std::complex<double> gate[4][4];
    return QuantumGate(type, target, control, gate);
}
