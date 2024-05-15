#include "helpers.h"
#include "gate.h"
#include "circuit.h"
#include "qubit_operator.h"
#include "sparse_tensor.h"

#include <stdexcept>
#include <algorithm>

namespace std {

template <> struct hash<Circuit> {
    std::size_t operator()(const Circuit& qc) const {
        std::string hash_value = "";

        for (const auto& gate : qc.gates()) {
            hash_value += gate.gate_id();
            hash_value += std::to_string(gate.control());
            hash_value += std::to_string(gate.target());
        }
        return hash<string>{}(hash_value);
    }
};
} // namespace std

void QubitOperator::add_term(std::complex<double> circ_coeff, const Circuit& circuit) {
    terms_.push_back(std::make_pair(circ_coeff, circuit));
}

void QubitOperator::add_op(const QubitOperator& qo) {
    for (const auto& term : qo.terms()) {
        terms_.push_back(std::make_pair(term.first, term.second));
    }
}

void QubitOperator::set_coeffs(const std::vector<std::complex<double>>& new_coeffs) {
    if (new_coeffs.size() != terms_.size()) {
        throw std::invalid_argument("number of new coefficients for quantum operator must equal ");
    }
    for (size_t l = 0; l < new_coeffs.size(); l++) {
        terms_[l].first = new_coeffs[l];
    }
}

void QubitOperator::mult_coeffs(const std::complex<double>& multiplier) {
    for (size_t l = 0; l < terms_.size(); l++) {
        terms_[l].first *= multiplier;
    }
}

void QubitOperator::order_terms() {
    simplify();
    std::sort(terms_.begin(), terms_.end(),
              [&](const std::pair<std::complex<double>, Circuit>& a,
                  const std::pair<std::complex<double>, Circuit>& b) {
                  int a_sz = a.second.gates().size();
                  int b_sz = b.second.gates().size();
                  // 1. sort by qb
                  for (int k = 0; k < std::min(a_sz, b_sz); k++) {
                      if (a.second.gates()[k].target() != b.second.gates()[k].target()) {
                          return (a.second.gates()[k].target() < b.second.gates()[k].target());
                      }
                  }
                  // 2. sort by gate id
                  for (int k = 0; k < std::min(a_sz, b_sz); k++) {
                      if (a.second.gates()[k].gate_id() != b.second.gates()[k].gate_id()) {
                          return (a.second.gates()[k].gate_id() < b.second.gates()[k].gate_id());
                      }
                  }
                  return (a.second.gates().size() < a.second.gates().size());
              });
}

void QubitOperator::canonical_order() {
    for (auto& term : terms_) {
        term.first *= term.second.canonicalize_pauli_circuit();
    }
}

void QubitOperator::simplify(bool combine_like_terms) {
    canonical_order();
    std::map<Circuit, std::complex<double>> uniqe_trms;
    for (const auto& term : terms_) {
        auto it = uniqe_trms.find(term.second);
        if (it == uniqe_trms.end()) {
            uniqe_trms.insert(std::make_pair(std::move(term.second), term.first));
        } else {
            it->second += term.first;
        }
    }
    terms_.clear();
    if (combine_like_terms) {
        for (const auto& uniqe_trm : uniqe_trms) {
            if (std::abs(uniqe_trm.second) > 1.0e-12) {
                terms_.emplace_back(uniqe_trm.second, uniqe_trm.first);
            }
        }
    } else {
        for (const auto& uniqe_trm : uniqe_trms) {
            terms_.emplace_back(uniqe_trm.second, uniqe_trm.first);
        }
    }
}

void QubitOperator::operator_product(const QubitOperator& rqo, bool pre_simplify,
                                     bool post_simplify) {
    if (pre_simplify) {
        simplify();
    }

    QubitOperator LR;
    for (auto& term_l : terms_) {
        for (auto& term_r : rqo.terms()) {
            Circuit temp_circ;
            temp_circ.add_circuit(term_l.second);
            temp_circ.add_circuit(term_r.second);
            LR.add_term(term_l.first * term_r.first, temp_circ);
        }
    }
    terms_ = std::move(LR.terms());

    if (post_simplify) {
        simplify();
    } else {
        canonical_order();
    }
}

const std::vector<std::pair<std::complex<double>, Circuit>>& QubitOperator::terms() const {
    return terms_;
}

bool QubitOperator::check_op_equivalence(QubitOperator qo, bool reorder) {
    if (reorder) {
        order_terms();
        qo.order_terms();
    }
    if (terms_.size() != qo.terms().size()) {
        return false;
    }
    for (size_t l = 0; l < terms_.size(); l++) {
        if (std::abs(terms_[l].first - qo.terms()[l].first) > 1.0e-10) {
            return false;
        }
        if (!(terms_[l].second == qo.terms()[l].second)) {
            return false;
        }
    }
    return true;
}

const SparseMatrix QubitOperator::sparse_matrix(size_t nqubit) const {
    SparseMatrix Rmat = SparseMatrix();
    if (terms_.empty()) {
        size_t nbasis = std::pow(2, nqubit);
        Rmat.make_identity(nbasis);
        return Rmat;
    }

    for (const auto& term : terms_) { // term -> [complex, Circuit]
        SparseMatrix Lmat = term.second.sparse_matrix(nqubit);
        Rmat.add(Lmat, term.first);
    }
    return Rmat;
}

std::string QubitOperator::str() const {
    std::vector<std::string> s;
    for (const auto& term : terms_) {
        s.push_back(to_string(term.first) + term.second.str());
    }
    return join(s, "\n");
}

size_t QubitOperator::num_qubits() const {
    size_t max = 0;
    for (const auto& summand : terms_) {
        max = std::max(max, summand.second.num_qubits());
    }
    return max;
}
