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

void SQOperator::caononical_order_single_term(std::pair< std::complex<double>, std::vector<size_t>>& term ){
    if((term.second.size() % 2) != 0){
        throw std::invalid_argument( "sq operator term must have equal number of anihilators and creators.");
    }
    int nbody = term.second.size() / 2.0;
    if (nbody >= 2) {
        auto term_temp = term;
        std::vector<int> a(nbody);
        std::iota(std::begin(a), std::end(a), 0);
        std::vector<int> b(nbody);
        std::iota(std::begin(b), std::end(b), 0);

        // get permutations for creators then reorder
        std::sort(a.begin(), a.end(),
            [&](const int& i, const int& j) {
                return (term_temp.second[i] > term_temp.second[j]);
            }
        );
        for (int ai=0; ai < nbody; ai++){
            term.second[ai] = term_temp.second[a[ai]];
        }
        if (permutive_sign_change(a)) { term.first *= -1.0; }

        // same as above but for anihilators
        std::sort(b.begin(), b.end(),
            [&](const int& i, const int& j) {
                return (term_temp.second[i+nbody] > term_temp.second[j+nbody]);
            }
        );
        for (int bi=0; bi < nbody; bi++){
            term.second[bi+nbody] = term_temp.second[b[bi]+nbody];
        }
        if (permutive_sign_change(b)) { term.first *= -1.0; }
    }
}

void SQOperator::canonical_order() {
    for (auto& term : terms_) {
        caononical_order_single_term(term);
    }
}

bool SQOperator::permutive_sign_change(std::vector<int> p) {
    std::vector<int> a(p.size());
    std::iota (std::begin(a), std::end(a), 0);
    size_t cnt = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        while (i != p[i]) {
            ++cnt;
            std::swap (a[i], a[p[i]]);
            std::swap (p[i], p[p[i]]);
        }
    }
    if(cnt % 2 == 0) {
        return false;
    } else {
        return true;
    }
}

// TODO: find out why size_t is printed as float
std::string SQOperator::str() const {
    std::vector<std::string> s;
    s.push_back("");
    for (const auto& term : terms_) {
        int nbody = term.second.size() / 2.0;
        s.push_back(to_string(term.first));
        s.push_back("(");
        for (int k=0; k<nbody; k++ ) {
            s.push_back(std::to_string(term.second[k]) + "^");
        }
        for (int k=nbody; k<2*nbody; k++ ) {
            s.push_back(std::to_string(term.second[k]));
        }
        s.push_back(")\n");
    }
    return join(s, " ");
}
