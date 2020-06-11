#include "helpers.h"
#include "quantum_gate.h"
#include "quantum_circuit.h"
#include "quantum_operator.h"

#include <stdexcept>

void QuantumOperator::add_term(std::complex<double> circ_coeff, const QuantumCircuit& circuit) {
    terms_.push_back(std::make_pair(circ_coeff, circuit));
}

void QuantumOperator::add_op(const QuantumOperator& qo) {
    for (const auto& term : qo.terms()) {
        terms_.push_back(std::make_pair(term.first, term.second));
    }
}

void QuantumOperator::set_coeffs(const std::vector<std::complex<double>>& new_coeffs) {
    if(new_coeffs.size() != terms_.size()){
        throw std::invalid_argument( "number of new coeficients for quantum operator must equal " );
    }
    for (size_t l = 0; l < new_coeffs.size(); l++){
        terms_[l].first = new_coeffs[l];
    }
}

// new funcitons here
void QuantumOperator::canonical_order() {
    for (auto& term : terms_) {
        term.first *= term.second.canonical_order();
    }
}

void QuantumOperator::add_unique_term(
    std::vector<std::pair<std::complex<double>, QuantumCircuit>>& uniqe_trms,
    const std::pair<std::complex<double>, QuantumCircuit>& term
) {
    bool not_in_unique = true;
    for (auto& uniqe_trm : uniqe_trms) {
        // if already in uniqe_trms, do a += anew
        if (uniqe_trm.second == term.second) {
            uniqe_trm.first += term.first;
            not_in_unique = false;
            break;
        }
    }
    if (not_in_unique) {
        uniqe_trms.push_back(term);
    }
}

void QuantumOperator::simplify() {
    // TODO: find a way with std::unordered_map
    // (see implementaion for SQOperator::simplify)
    canonical_order();
    std::vector<std::pair<std::complex<double>, QuantumCircuit>> uniqe_trms;
    for (auto& term : terms_) {
        add_unique_term(uniqe_trms, term);
    }
    terms_ = std::move(uniqe_trms);
}

const std::vector<std::pair<std::complex<double>, QuantumCircuit>>& QuantumOperator::terms() const {
    return terms_;
}

std::string QuantumOperator::str() const {
    std::vector<std::string> s;
    for (const auto& term : terms_) {
        s.push_back(to_string(term.first) + term.second.str());
    }
    return join(s, "\n");
}
