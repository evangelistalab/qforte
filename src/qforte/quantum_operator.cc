#include "helpers.h"
#include "quantum_gate.h"
#include "quantum_circuit.h"
#include "quantum_operator.h"

#include <stdexcept>

namespace std {

    template <>
    struct hash<QuantumCircuit>{
        std::size_t operator()(const QuantumCircuit& qc) const {
            std::string hash_value = "";

            for (const auto& gate : qc.gates()){
                hash_value += gate.gate_id();
                hash_value += std::to_string(gate.control());
                hash_value += std::to_string(gate.target());
            }
            return hash<string>{}(hash_value);
        }
    };
}

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

void QuantumOperator::mult_coeffs(const std::complex<double>& multiplier) {
    for (size_t l = 0; l < terms_.size(); l++){
        terms_[l].first *= multiplier;
    }
}

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

void QuantumOperator::map_simplify() {
    canonical_order();
    std::unordered_map<QuantumCircuit, std::complex<double>> uniqe_trms;
    for (const auto& term : terms_) {
        if ( uniqe_trms.find(term.second) == uniqe_trms.end() ) {
            uniqe_trms.insert(std::make_pair(term.second, term.first));
        } else {
            uniqe_trms[term.second] += term.first;
        }
    }
    terms_.clear();
    for (const auto &uniqe_trm : uniqe_trms){
        if (std::abs(uniqe_trm.second) > 0.0) {
            terms_.push_back(std::make_pair(uniqe_trm.second, uniqe_trm.first));
        }
    }
}

void QuantumOperator::simplify() {
    canonical_order();
    std::vector<std::pair<std::complex<double>, QuantumCircuit>> uniqe_trms;
    for (auto& term : terms_) {
        add_unique_term(uniqe_trms, term);
    }
    // TODO: need to account for the removal of terms with coeff = 0.0
    terms_ = std::move(uniqe_trms);
}

void QuantumOperator::join_operator(QuantumOperator& rqo, bool simplify_lop_rop ) {
    if(simplify_lop_rop){
        map_simplify();
        rqo.map_simplify();
    }

    QuantumOperator LR;
    for (auto& term_l : terms_) {
        for (auto& term_r : rqo.terms()){
            QuantumCircuit temp_circ;
            temp_circ.add_circuit(term_l.second);
            temp_circ.add_circuit(term_r.second);
            LR.add_term(term_l.first * term_r.first, temp_circ);
        }
    }
    terms_ = std::move(LR.terms());
    map_simplify();
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
