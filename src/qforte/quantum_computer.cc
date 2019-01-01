#include <map>

#include "fmt/format.h"

#include "quantum_gate.h"
#include "quantum_computer.h"

std::string Basis::str(size_t nqubit) const {
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

std::vector<std::string> QuantumCircuit::str() const {
    std::vector<std::string> s;
    for (const auto& gate : gates_) {
        s.push_back(gate.str());
    }
    return s;
}

QuantumComputer::QuantumComputer(int nqubit) : nqubit_(nqubit) {
    nbasis_ = std::pow(2, nqubit_);
    basis_.assign(nbasis_, Basis());
    coeff_.assign(nbasis_, 0.0);
    new_coeff_.assign(nbasis_, 0.0);
    for (size_t i = 0; i < nbasis_; i++) {
        basis_[i] = Basis(i);
    }
    coeff_[0] = 1.;
}

std::complex<double> QuantumComputer::coeff(const Basis& basis) { return coeff_[basis.add()]; }

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

    coeff_ = new_coeff_;
    std::fill(new_coeff_.begin(), new_coeff_.end(), 0.0);
}

void QuantumComputer::apply_1qubit_gate(const QuantumGate& qg) {
    size_t target = qg.target();
    const auto& gate = qg.gate();

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            auto op_i_j = gate[i][j];
            for (const Basis& basis_I : basis_) {
                if (basis_I.get_bit(target) == j) {
                    Basis basis_J = basis_I;
                    basis_J.set_bit(target, i);
                    new_coeff_[basis_J.add()] += op_i_j * coeff_[basis_I.add()];
                }
            }
        }
    }
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
