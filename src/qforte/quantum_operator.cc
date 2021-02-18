#include "helpers.h"
#include "quantum_gate.h"
#include "quantum_circuit.h"
#include "quantum_operator.h"

#include <stdexcept>
#include <algorithm>

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

void QuantumOperator::order_terms() {
    simplify();
    std::sort(terms_.begin(), terms_.end(),
        [&](const std::pair<std::complex<double>, QuantumCircuit>& a,
            const std::pair<std::complex<double>, QuantumCircuit>& b) {
                int a_sz = a.second.gates().size();
                int b_sz = b.second.gates().size();
                // 1. sort by qb
                for (int k=0; k<std::min(a_sz, b_sz); k++){
                    if( a.second.gates()[k].target() != b.second.gates()[k].target()){
                        return (a.second.gates()[k].target() < b.second.gates()[k].target());
                    }
                }
                // 2. sort by gate id
                for (int k=0; k<std::min(a_sz, b_sz); k++){
                    if(a.second.gates()[k].gate_id() != b.second.gates()[k].gate_id()){
                        return (a.second.gates()[k].gate_id() < b.second.gates()[k].gate_id());
                    }
                }
                return (a.second.gates().size() < a.second.gates().size());
        }
    );
}

void QuantumOperator::canonical_order() {
    for (auto& term : terms_) {
        term.first *= term.second.canonical_order();
    }
}

void QuantumOperator::simplify() {
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
        if (std::abs(uniqe_trm.second) > 1.0e-16) {
            terms_.push_back(std::make_pair(uniqe_trm.second, uniqe_trm.first));
        }
    }
}

void QuantumOperator::join_operator(const QuantumOperator& rqo, bool simplify_lop ) {
    if(simplify_lop){
        simplify();
        // rqo.simplify();
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
    simplify();
}

void QuantumOperator::join_operator_lazy(const QuantumOperator& rqo) {
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
    canonical_order();
}

const std::vector<std::pair<std::complex<double>, QuantumCircuit>>& QuantumOperator::terms() const {
    return terms_;
}

bool QuantumOperator::check_op_equivalence(QuantumOperator qo, bool reorder) {
    if(reorder){
        order_terms();
        qo.order_terms();
    }
    if (terms_.size() != qo.terms().size()){
        return false;
    }
    for (size_t l = 0; l < terms_.size(); l++){
        if(std::abs(terms_[l].first-qo.terms()[l].first) > 1.0e-10){
            return false;
        }
        if (!(terms_[l].second == qo.terms()[l].second)) {
            return false;
        }
    }
    return true;
}

std::string QuantumOperator::str() const {
    std::vector<std::string> s;
    for (const auto& term : terms_) {
        s.push_back(to_string(term.first) + term.second.str());
    }
    return join(s, "\n");
}
