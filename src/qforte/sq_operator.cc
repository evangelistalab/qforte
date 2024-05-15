#include <algorithm>
#include <stdexcept>
#include <tuple>

#include "helpers.h"
#include "gate.h"
#include "circuit.h"
#include "qubit_operator.h"
#include "sq_operator.h"

void SQOperator::add_term(std::complex<double> circ_coeff, const std::vector<size_t>& cre_ops,
                          const std::vector<size_t>& ann_ops) {
    terms_.push_back(std::make_tuple(circ_coeff, cre_ops, ann_ops));
}

void SQOperator::add_op(const SQOperator& qo) {
    terms_.insert(terms_.end(), qo.terms().begin(), qo.terms().end());
}

void SQOperator::set_coeffs(const std::vector<std::complex<double>>& new_coeffs) {
    if (new_coeffs.size() != terms_.size()) {
        throw std::invalid_argument("number of new coefficients for quantum operator must equal ");
    }
    for (auto l = 0; l < new_coeffs.size(); l++) {
        std::get<0>(terms_[l]) = new_coeffs[l];
    }
}

void SQOperator::mult_coeffs(const std::complex<double>& multiplier) {
    for (auto& term : terms_) {
        std::get<0>(term) *= multiplier;
    }
}

const std::vector<std::tuple<std::complex<double>, std::vector<size_t>, std::vector<size_t>>>&
SQOperator::terms() const {
    return terms_;
}

int SQOperator::canonicalize_helper(std::vector<size_t>& op_list) const {
    auto temp_op = op_list;
    auto length = temp_op.size();
    {
        std::vector<int> temp(length);
        std::iota(std::begin(temp), std::end(temp), 0);
        std::sort(temp.begin(), temp.end(),
                  [&](const int& i, const int& j) { return (temp_op[i] > temp_op[j]); });
        for (int i = 0; i < length; i++) {
            op_list[i] = temp_op[temp[i]];
        }
        return (permutation_phase(temp)) ? -1 : 1;
    }
}

void SQOperator::canonical_order_single_term(
    std::tuple<std::complex<double>, std::vector<size_t>, std::vector<size_t>>& term) {
    std::get<0>(term) *= canonicalize_helper(std::get<1>(term));
    std::get<0>(term) *= canonicalize_helper(std::get<2>(term));
}

void SQOperator::canonical_order() {
    for (auto& term : terms_) {
        canonical_order_single_term(term);
    }
}

void SQOperator::simplify() {
    canonical_order();
    std::map<std::pair<std::vector<size_t>, std::vector<size_t>>, std::complex<double>>
        unique_terms;
    for (const auto& term : terms_) {
        auto pair = std::make_pair(std::get<1>(term), std::get<2>(term));
        if (unique_terms.find(pair) == unique_terms.end()) {
            unique_terms.insert(std::make_pair(pair, std::get<0>(term)));
        } else {
            unique_terms[pair] += std::get<0>(term);
        }
    }
    terms_.clear();
    for (const auto& unique_term : unique_terms) {
        if (std::abs(unique_term.second) > 1.0e-12) {
            terms_.push_back(std::make_tuple(unique_term.second, unique_term.first.first,
                                             unique_term.first.second));
        }
    }
}

bool SQOperator::permutation_phase(std::vector<int> p) const {
    std::vector<int> a(p.size());
    std::iota(std::begin(a), std::end(a), 0);
    size_t cnt = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        while (i != p[i]) {
            ++cnt;
            std::swap(a[i], a[p[i]]);
            std::swap(p[i], p[p[i]]);
        }
    }
    if (cnt % 2 == 0) {
        return false;
    } else {
        return true;
    }
}

void SQOperator::jw_helper(QubitOperator& holder, const std::vector<size_t>& operators,
                           bool creator, bool qubit_excitation) const {
    std::complex<double> halfi(0.0, 0.5);
    if (creator) {
        halfi *= -1;
    };

    for (const auto& sq_op : operators) {
        QubitOperator temp;
        Circuit Xcirc;
        Circuit Ycirc;

        if (not qubit_excitation) {
            for (int k = 0; k < sq_op; k++) {
                Xcirc.add_gate(make_gate("Z", k, k));
                Ycirc.add_gate(make_gate("Z", k, k));
            }
        }

        Xcirc.add_gate(make_gate("X", sq_op, sq_op));
        Ycirc.add_gate(make_gate("Y", sq_op, sq_op));
        temp.add_term(0.5, Xcirc);
        temp.add_term(halfi, Ycirc);

        if (holder.terms().size() == 0) {
            holder.add_op(temp);
        } else {
            holder.operator_product(temp);
        }
    }
}

QubitOperator SQOperator::jw_transform(bool qubit_excitation) {
    /// The simplify() function also brings second-quantized operators
    /// to normal order. This also ensures the 1-to-1 mapping between
    /// second-quantized and qubit operators when qubit_excitation=True
    simplify();
    QubitOperator qo;

    for (const auto& fermion_operator : terms_) {
        auto cre_length = std::get<1>(fermion_operator).size();
        auto ann_length = std::get<2>(fermion_operator).size();

        if (cre_length == 0 && ann_length == 0) {
            // Scalars need special logic.
            Circuit scalar_circ;
            QubitOperator scalar_op;
            scalar_op.add_term(std::get<0>(fermion_operator), scalar_circ);
            qo.add_op(scalar_op);
            continue;
        }

        QubitOperator temp1;
        jw_helper(temp1, std::get<1>(fermion_operator), true, qubit_excitation);
        jw_helper(temp1, std::get<2>(fermion_operator), false, qubit_excitation);

        temp1.mult_coeffs(std::get<0>(fermion_operator));
        qo.add_op(temp1);
    }

    qo.simplify();

    return qo;
}

std::string SQOperator::str() const {
    std::vector<std::string> s;
    s.push_back("");
    for (const auto& term : terms_) {
        s.push_back(to_string(std::get<0>(term)));
        s.push_back("(");
        for (auto k : std::get<1>(term)) {
            s.push_back(std::to_string(k) + "^");
        }
        for (auto k : std::get<2>(term)) {
            s.push_back(std::to_string(k));
        }
        s.push_back(")\n");
    }
    return join(s, " ");
}
