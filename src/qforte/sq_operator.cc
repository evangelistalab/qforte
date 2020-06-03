#include "helpers.h"
// #include "quantum_gate.h"
// #include "quantum_circuit.h"
// #include "quantum_operator.h"
#include "sq_operator.h"

#include <stdexcept>

void SQOperator::add_term(std::complex<double> circ_coeff, const std::vector<size_t>& ac_ops) {
    terms_.push_back(std::make_pair(circ_coeff, ac_ops));
}

void SQOperator::add_op(const SQOperator& qo) {
    for (const auto& term : qo.terms()) {
        terms_.push_back(std::make_pair(term.first, term.second));
    }
}

void SQOperator::set_coeffs(const std::vector<std::complex<double>>& new_coeffs) {
    if(new_coeffs.size() != terms_.size()){
        throw std::invalid_argument( "number of new coeficients for quantum operator must equal " );
    }
    for (size_t l = 0; l < new_coeffs.size(); l++){
        terms_[l].first = new_coeffs[l];
    }
}

const std::vector<std::pair<std::complex<double>, std::vector<size_t>>>& SQOperator::terms() const {
    return terms_;
}

std::string SQOperator::str() const {
    std::vector<std::string> s;
    for (const auto& term : terms_) {
        s.push_back(to_string(term.first) + term.second.str());
    }
    return join(s, "\n");
}
