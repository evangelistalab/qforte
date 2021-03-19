#include <algorithm>

#include "helpers.h"
#include "quantum_gate.h"
#include "quantum_circuit.h"
#include "quantum_operator.h"
#include "sq_operator.h"

#include <stdexcept>
#include <algorithm>

void SQOperator::add_term(std::complex<double> circ_coeff, const std::vector<size_t>& ac_ops) {
    if ((ac_ops.size() % 2) != 0) throw std::invalid_argument("Term must have N creators and N annihilators, but received odd number of orbital indices");
    terms_.push_back(std::make_pair(circ_coeff, ac_ops));
}

void SQOperator::add_op(const SQOperator& qo) {
    for (const auto& term : qo.terms()) {
        terms_.push_back(std::make_pair(term.first, term.second));
    }
}

void SQOperator::set_coeffs(const std::vector<std::complex<double>>& new_coeffs) {
    if(new_coeffs.size() != terms_.size()){
        throw std::invalid_argument( "number of new coefficients for quantum operator must equal " );
    }
    for (size_t l = 0; l < new_coeffs.size(); l++){
        terms_[l].first = new_coeffs[l];
    }
}

void SQOperator::mult_coeffs(const std::complex<double>& multiplier) {
    for (size_t l = 0; l < terms_.size(); l++){
        terms_[l].first *= multiplier;
    }
}

const std::vector<std::pair<std::complex<double>, std::vector<size_t>>>& SQOperator::terms() const {
    return terms_;
}

void SQOperator::canonical_order_single_term(std::pair< std::complex<double>, std::vector<size_t>>& term ){
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
        // same as above but for annihilators
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
        canonical_order_single_term(term);
    }
}

void SQOperator::simplify() {
    canonical_order();
    std::map<std::vector<size_t>, std::complex<double>> uniqe_trms;
    for (const auto& term : terms_) {
        if ( uniqe_trms.find(term.second) == uniqe_trms.end() ) {
            uniqe_trms.insert(std::make_pair(term.second, term.first));
        } else {
            uniqe_trms[term.second] += term.first;
        }
    }
    terms_.clear();
    for (const auto &uniqe_trm : uniqe_trms){
        if (std::abs(uniqe_trm.second) > 1.0e-16){
            terms_.push_back(std::make_pair(uniqe_trm.second, uniqe_trm.first));
        }
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

QuantumOperator SQOperator::jw_transform() {
    std::complex<double> halfi(0.0, 0.5);
    simplify(); // This isn't needed for the logic - just an efficiency optimization.
    QuantumOperator qo;

    for (const auto& fermion_operator : terms_) {
        if((fermion_operator.second.size() % 2) != 0){
            throw std::invalid_argument( "sq operator term must have equal number of annihilators and creators. This error should be unreachable - debugging QForte needed.");
        }
        int nbody = fermion_operator.second.size() / 2.0;

        if (nbody == 0) {
            // Scalars need special logic.
            QuantumCircuit scalar_circ;
            QuantumOperator scalar_op;
            scalar_op.add_term(term.first, scalar_circ);
            qo.add_op(scalar_op);
            continue;
        }
        QuantumOperator temp1;
        for (int ai=0; ai<2*nbody; ai++) {
            QuantumOperator temp2;
            // Our qubit operator is (X +/- i Y) phase-factor-Z gates.
            // We need two circuits in our linear combination.
            QuantumCircuit Xcirc;
            QuantumCircuit Ycirc;

            // Z gates for the phase factor
            for(int k=0; k<term.second[ai]; k++){
                Xcirc.add_gate(make_gate("Z", k, k));
                Ycirc.add_gate(make_gate("Z", k, k));
            }

            Xcirc.add_gate(make_gate("X", term.second[ai], term.second[ai]));
            Ycirc.add_gate(make_gate("Y", term.second[ai], term.second[ai]));
            temp2.add_term(0.5, Xcirc);

            // TODO: Remove below code's assumptions of vacuum-normal and particle-conserving.
            // Will certainly require changing the innards of SQOperator.
            if(ai < nbody){
                // We have an annihilation operator.
                temp2.add_term(-halfi, Ycirc);
            } else {
                // We have a creation operator.
                temp2.add_term(halfi, Ycirc);
            }

            if (ai == 0) {
                temp1.add_op(temp2);
            } else {
                temp1.operator_product(temp2);
            }
        }
        temp1.mult_coeffs(term.first);
        qo.add_op(temp1);
    }
    qo.simplify();
    // Consider also standard ordering these?
    return qo;
}

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
