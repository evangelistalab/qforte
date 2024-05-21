#include "helpers.h"
#include "gate.h"
#include "circuit.h"
#include "qubit_operator.h"
#include "sq_operator.h"
#include "sq_op_pool.h"
#include "qubit_op_pool.h"

#include <stdexcept>
#include <algorithm>
#include <iostream>

void QubitOpPool::add_term(std::complex<double> coeff, const QubitOperator& sq_op) {
    terms_.push_back(std::make_pair(coeff, sq_op));
}

void QubitOpPool::set_coeffs(const std::vector<std::complex<double>>& new_coeffs) {
    if (new_coeffs.size() != terms_.size()) {
        throw std::invalid_argument(
            "Number of new coefficients for quantum op pool must equal number of terms in pool.");
    }
    for (size_t l = 0; l < new_coeffs.size(); l++) {
        terms_[l].first = new_coeffs[l];
    }
}

void QubitOpPool::set_op_coeffs(const std::vector<std::complex<double>>& new_coeffs) {
    for (size_t I = 0; I < terms_.size(); I++) {
        if (new_coeffs.size() != terms_[I].second.terms().size()) {
            throw std::invalid_argument("Number of new coeficients for quantum operator must equal "
                                        "number of terim in operator.");
        }
        terms_[I].second.set_coeffs(new_coeffs);
    }
}

void QubitOpPool::set_terms(
    std::vector<std::pair<std::complex<double>, QubitOperator>>& new_terms) {
    // TODO: consider clearing terms_ when this fuction is called
    for (const auto& term : new_terms) {
        terms_.push_back(term);
    }
}

const std::vector<std::pair<std::complex<double>, QubitOperator>>& QubitOpPool::terms() const {
    return terms_;
}

void QubitOpPool::square(bool upper_triangle_only) {
    std::vector<std::pair<std::complex<double>, QubitOperator>> temp_terms;
    if (upper_triangle_only) {
        // consider only I -> IJ, where J > I
        for (int I = 0; I < terms_.size(); I++) {
            for (int J = I; J < terms_.size(); J++) {
                QubitOperator IJ;
                IJ.add_op(terms_[I].second);
                IJ.operator_product(terms_[J].second, false);
                IJ.simplify();
                temp_terms.push_back(
                    std::make_pair(std::conj(terms_[I].first) * terms_[J].first, IJ));
            }
        }
    } else {
        // consider all I -> IJ
        for (auto& I : terms_) {
            for (auto& J : terms_) {
                QubitOperator IJ;
                IJ.add_op(I.second);
                IJ.operator_product(J.second, false);
                IJ.simplify();
                temp_terms.push_back(std::make_pair(std::conj(I.first) * J.first, IJ));
            }
        }
    }
    terms_ = std::move(temp_terms);
}

void QubitOpPool::join_op_from_right(const QubitOperator& q_op) {
    for (auto& term : terms_) {
        term.second.operator_product(q_op, false);
        term.second.simplify();
    }
}

void QubitOpPool::join_op_from_right_lazy(const QubitOperator& q_op) {
    for (auto& term : terms_) {
        term.second.operator_product(q_op, false, false);
    }
}

void QubitOpPool::join_op_from_left(const QubitOperator& q_op) {
    std::vector<std::pair<std::complex<double>, QubitOperator>> temp_terms;
    for (const auto& term : terms_) {
        QubitOperator temp_op;
        temp_op.add_op(q_op);
        temp_op.operator_product(term.second, false);
        temp_op.simplify();
        temp_terms.push_back(std::make_pair(term.first, temp_op));
    }
    terms_ = std::move(temp_terms);
}

void QubitOpPool::join_as_commutator(const QubitOperator& q_op) {
    std::vector<std::pair<std::complex<double>, QubitOperator>> temp_terms;
    for (const auto& term : terms_) {
        // build HAm
        QubitOperator HAm;
        HAm.add_op(q_op);
        HAm.operator_product(term.second, false);

        // build -AmH
        QubitOperator AmH;
        AmH.add_op(term.second);
        AmH.operator_product(q_op, false);
        AmH.mult_coeffs(-1.0);

        HAm.add_op(AmH);
        HAm.simplify();
        temp_terms.push_back(std::make_pair(term.first, HAm));
    }
    terms_ = std::move(temp_terms);
}

void QubitOpPool::fill_pool(std::string pool_type, const size_t nqb) {
    if (pool_type == "complete_qubit") {
        std::map<std::string, std::string> paulis = {
            {"0", "I"}, {"1", "X"}, {"2", "Y"}, {"3", "Z"}};
        int nterms = static_cast<int>(std::pow(4, nqb));

        for (int I = 0; I < nterms; I++) {
            QubitOperator AI;
            Circuit aI;
            auto paulistr = pauli_idx_str(to_base4(I), nqb);
            if (paulistr.length() != nqb) {
                throw std::invalid_argument("paulistr.length() != nqb");
            }

            for (size_t k = 0; k < nqb; k++) {
                if (paulistr.substr(k, 1) != "0") {
                    aI.add_gate(make_gate(paulis[paulistr.substr(k, 1)], k, k));
                }
            }
            AI.add_term(1.0, aI);
            add_term(1.0, AI);
        }
    } else if (pool_type == "cqoy") {
        std::map<std::string, std::string> paulis = {
            {"0", "I"}, {"1", "X"}, {"2", "Y"}, {"3", "Z"}};
        int nterms = static_cast<int>(std::pow(4, nqb));

        for (int I = 0; I < nterms; I++) {
            QubitOperator AI;
            Circuit aI;
            auto paulistr = pauli_idx_str(to_base4(I), nqb);
            if (paulistr.length() != nqb) {
                throw std::invalid_argument("paulistr.length() != nqb");
            }
            int nygates = 0;
            for (size_t k = 0; k < nqb; k++) {
                if (paulistr.substr(k, 1) == "2") {
                    nygates++;
                }
                if (paulistr.substr(k, 1) != "0") {
                    aI.add_gate(make_gate(paulis[paulistr.substr(k, 1)], k, k));
                }
            }
            if (nygates % 2 != 0) {
                AI.add_term(1.0, aI);
                add_term(1.0, AI);
            }
        }
    } else {
        throw std::invalid_argument("Invalid pool_type specified.");
    }
}

std::string QubitOpPool::str() const {
    std::vector<std::string> s;
    s.push_back("");
    int counter = 0;
    for (const auto& term : terms_) {
        s.push_back("----->");
        s.push_back(std::to_string(counter));
        s.push_back("<-----\n");
        s.push_back(to_string(term.first));
        s.push_back("[\n");
        s.push_back(term.second.str());
        s.push_back("\n");
        s.push_back("]\n\n");
        counter++;
    }
    return join(s, " ");
}

std::string QubitOpPool::to_base4(int I) {
    std::string convert_str = "0123456789";
    if (I < 4) {
        return convert_str.substr(I, 1);
    } else {
        return to_base4(std::floor(I / 4)) + convert_str.substr(I % 4, 1);
    }
}

std::string QubitOpPool::pauli_idx_str(std::string I_str, int nqb) {
    std::string res;
    for (int i = 0; i < nqb - I_str.length(); i++) {
        res.append("0");
    }
    return res + I_str;
}
