#include "helpers.h"
#include "quantum_gate.h"
#include "quantum_circuit.h"
#include "quantum_operator.h"

void QuantumOperator::add_term(std::complex<double> circ_coeff, const QuantumCircuit& circuit) {
    terms_.push_back(std::make_pair(circ_coeff, circuit));
}

const std::vector<std::pair<std::complex<double>, QuantumCircuit>>& QuantumOperator::terms() const {
    return terms_;
}

std::string QuantumOperator::str() const {
    std::vector<std::string> s;
    for (const auto& term : terms_) {
        s.push_back(to_string(term.first) + term.second.str());
    }
    return join(s, " ");
}
