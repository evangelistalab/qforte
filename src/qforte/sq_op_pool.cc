#include "helpers.h"
#include "gate.h"
#include "circuit.h"
#include "qubit_operator.h"
#include "sq_operator.h"
#include "qubit_op_pool.h"
#include "sq_op_pool.h"
#include "find_irrep.h"
#include "qubit_basis.h"

#include <stdexcept>
#include <algorithm>
#include <bitset>

void SQOpPool::add_term(std::complex<double> coeff, const SQOperator& sq_op) {
    terms_.push_back(std::make_pair(coeff, sq_op));
}

void SQOpPool::set_coeffs(const std::vector<std::complex<double>>& new_coeffs) {
    if (new_coeffs.size() != terms_.size()) {
        throw std::invalid_argument("Number of new coefficients for quantum operator must equal.");
    }
    for (size_t l = 0; l < new_coeffs.size(); l++) {
        terms_[l].first = new_coeffs[l];
    }
}

const std::vector<std::pair<std::complex<double>, SQOperator>>& SQOpPool::terms() const {
    return terms_;
}

void SQOpPool::set_orb_spaces(const std::vector<int>& ref,
                              const std::vector<size_t>& orb_irreps_to_int) {
    // compute integer representing reference determiant
    ref_int_ = 0;
    int multiplier = 1;
    for (const auto& occupancy : ref) {
        ref_int_ += occupancy * multiplier;
        multiplier = multiplier << 1;
    }

    // set orbital spaces
    n_spinorb_ = ref.size();
    if (n_spinorb_ % 2 != 0) {
        throw std::invalid_argument("The total number of spinorbitals must be even!");
    }

    n_occ_alpha_ = 0;
    n_occ_beta_ = 0;
    n_vir_alpha_ = 0;
    n_vir_beta_ = 0;

    bool is_alpha = true;

    for (const auto& occupancy : ref) {
        if (is_alpha) {
            n_occ_alpha_ += occupancy;
            n_vir_alpha_ += occupancy ^ 1;
        } else {
            n_occ_beta_ += occupancy;
            n_vir_beta_ += occupancy ^ 1;
        }
        is_alpha = !is_alpha;
    }

    if (orb_irreps_to_int.empty()) {
        orb_irreps_to_int_ = std::vector<size_t>(n_spinorb_ / 2, 0);
    } else {
        orb_irreps_to_int_ = orb_irreps_to_int;
    }
}

QubitOpPool SQOpPool::get_qubit_op_pool() {
    QubitOpPool A;
    for (auto& term : terms_) {
        // QubitOperator a = term.second.jw_transform();
        // a.mult_coeffs(term.first);
        A.add_term(term.first, term.second.jw_transform());
    }
    return A;
}

QubitOperator SQOpPool::get_qubit_operator(const std::string& order_type, bool combine_like_terms,
                                           bool qubit_excitations) {
    QubitOperator parent;

    if (order_type == "unique_lex") {
        for (auto& term : terms_) {
            auto child = term.second.jw_transform(qubit_excitations);
            child.mult_coeffs(term.first);
            parent.add_op(child);
        }
        // TODO: analyze ordering here, eliminating simplify will place commuting
        // terms closer together but may introduce redundancy.
        parent.simplify();
        parent.order_terms();
    } else if (order_type == "commuting_grp_lex") {
        for (auto& term : terms_) {
            auto child = term.second.jw_transform(qubit_excitations);
            child.mult_coeffs(term.first);
            child.simplify(combine_like_terms = combine_like_terms);
            child.order_terms();
            parent.add_op(child);
        }
    } else {
        throw std::invalid_argument("Invalid order_type specified.");
    }
    return parent;
}

void SQOpPool::fill_pool(std::string pool_type) {
    if (pool_type == "GSD") {
        size_t norb = n_spinorb_ / 2;
        for (size_t i = 0; i < norb; i++) {
            size_t ia = 2 * i;
            size_t ib = 2 * i + 1;
            for (size_t a = i; a < norb; a++) {
                size_t aa = 2 * a;
                size_t ab = 2 * a + 1;

                if (!find_irrep(orb_irreps_to_int_, std::vector<size_t>{ia, aa})) {

                    if (aa != ia) {
                        SQOperator temp1a;
                        temp1a.add_term(+1.0, {aa}, {ia});
                        temp1a.add_term(-1.0, {ia}, {aa});
                        temp1a.simplify();
                        if (temp1a.terms().size() > 0) {
                            add_term(1.0, temp1a);
                        }
                    }

                    if (ab != ib) {
                        SQOperator temp1b;
                        temp1b.add_term(+1.0, {ab}, {ib});
                        temp1b.add_term(-1.0, {ib}, {ab});
                        temp1b.simplify();
                        if (temp1b.terms().size() > 0) {
                            add_term(1.0, temp1b);
                        }
                    }
                }
            }
        }

        std::vector<std::vector<size_t>> uniqe_2bdy;
        std::vector<std::vector<size_t>> adjnt_2bdy;

        for (size_t i = 0; i < norb; i++) {
            size_t ia = 2 * i;
            size_t ib = 2 * i + 1;
            for (size_t j = i; j < norb; j++) {
                size_t ja = 2 * j;
                size_t jb = 2 * j + 1;
                for (size_t a = 0; a < norb; a++) {
                    size_t aa = 2 * a;
                    size_t ab = 2 * a + 1;
                    for (size_t b = a; b < norb; b++) {
                        size_t ba = 2 * b;
                        size_t bb = 2 * b + 1;

                        if (!find_irrep(orb_irreps_to_int_, std::vector<size_t>{ia, ja, aa, ba})) {

                            if ((aa != ba) && (ia != ja)) {
                                SQOperator temp2aaaa;
                                temp2aaaa.add_term(+1.0, {aa, ba}, {ia, ja});
                                temp2aaaa.add_term(-1.0, {ja, ia}, {ba, aa});
                                temp2aaaa.simplify();
                                if (temp2aaaa.terms().size() > 0) {
                                    std::vector<size_t> vtemp{std::get<1>(temp2aaaa.terms()[0])[0],
                                                              std::get<1>(temp2aaaa.terms()[0])[1],
                                                              std::get<2>(temp2aaaa.terms()[0])[0],
                                                              std::get<2>(temp2aaaa.terms()[0])[1]};
                                    std::vector<size_t> vadjt{std::get<1>(temp2aaaa.terms()[1])[0],
                                                              std::get<1>(temp2aaaa.terms()[1])[1],
                                                              std::get<2>(temp2aaaa.terms()[1])[0],
                                                              std::get<2>(temp2aaaa.terms()[1])[1]};
                                    if ((std::find(uniqe_2bdy.begin(), uniqe_2bdy.end(), vtemp) ==
                                         uniqe_2bdy.end())) {
                                        if ((std::find(adjnt_2bdy.begin(), adjnt_2bdy.end(),
                                                       vtemp) == adjnt_2bdy.end())) {
                                            uniqe_2bdy.push_back(vtemp);
                                            adjnt_2bdy.push_back(vadjt);
                                            add_term(1.0, temp2aaaa);
                                        }
                                    }
                                }
                            }

                            if ((ab != bb) && (ib != jb)) {
                                SQOperator temp2bbbb;
                                temp2bbbb.add_term(+1.0, {ab, bb}, {ib, jb});
                                temp2bbbb.add_term(-1.0, {jb, ib}, {bb, ab});
                                temp2bbbb.simplify();
                                if (temp2bbbb.terms().size() > 0) {
                                    std::vector<size_t> vtemp{std::get<1>(temp2bbbb.terms()[0])[0],
                                                              std::get<1>(temp2bbbb.terms()[0])[1],
                                                              std::get<2>(temp2bbbb.terms()[0])[0],
                                                              std::get<2>(temp2bbbb.terms()[0])[1]};
                                    std::vector<size_t> vadjt{std::get<1>(temp2bbbb.terms()[1])[0],
                                                              std::get<1>(temp2bbbb.terms()[1])[1],
                                                              std::get<2>(temp2bbbb.terms()[1])[0],
                                                              std::get<2>(temp2bbbb.terms()[1])[1]};
                                    if ((std::find(uniqe_2bdy.begin(), uniqe_2bdy.end(), vtemp) ==
                                         uniqe_2bdy.end())) {
                                        if ((std::find(adjnt_2bdy.begin(), adjnt_2bdy.end(),
                                                       vtemp) == adjnt_2bdy.end())) {
                                            uniqe_2bdy.push_back(vtemp);
                                            adjnt_2bdy.push_back(vadjt);
                                            add_term(1.0, temp2bbbb);
                                        }
                                    }
                                }
                            }

                            if ((aa != bb) && (ia != jb)) {
                                SQOperator temp2abab;
                                temp2abab.add_term(+1.0, {aa, bb}, {ia, jb});
                                temp2abab.add_term(-1.0, {jb, ia}, {bb, aa});
                                temp2abab.simplify();
                                if (temp2abab.terms().size() > 0) {
                                    std::vector<size_t> vtemp{std::get<1>(temp2abab.terms()[0])[0],
                                                              std::get<1>(temp2abab.terms()[0])[1],
                                                              std::get<2>(temp2abab.terms()[0])[0],
                                                              std::get<2>(temp2abab.terms()[0])[1]};
                                    std::vector<size_t> vadjt{std::get<1>(temp2abab.terms()[1])[0],
                                                              std::get<1>(temp2abab.terms()[1])[1],
                                                              std::get<2>(temp2abab.terms()[1])[0],
                                                              std::get<2>(temp2abab.terms()[1])[1]};
                                    if ((std::find(uniqe_2bdy.begin(), uniqe_2bdy.end(), vtemp) ==
                                         uniqe_2bdy.end())) {
                                        if ((std::find(adjnt_2bdy.begin(), adjnt_2bdy.end(),
                                                       vtemp) == adjnt_2bdy.end())) {
                                            uniqe_2bdy.push_back(vtemp);
                                            adjnt_2bdy.push_back(vadjt);
                                            add_term(1.0, temp2abab);
                                        }
                                    }
                                }
                            }

                            if ((ab != ba) && (ib != ja)) {
                                SQOperator temp2baba;
                                temp2baba.add_term(+1.0, {ab, ba}, {ib, ja});
                                temp2baba.add_term(-1.0, {ja, ib}, {ba, ab});
                                temp2baba.simplify();
                                if (temp2baba.terms().size() > 0) {
                                    std::vector<size_t> vtemp{std::get<1>(temp2baba.terms()[0])[0],
                                                              std::get<1>(temp2baba.terms()[0])[1],
                                                              std::get<2>(temp2baba.terms()[0])[0],
                                                              std::get<2>(temp2baba.terms()[0])[1]};
                                    std::vector<size_t> vadjt{std::get<1>(temp2baba.terms()[1])[0],
                                                              std::get<1>(temp2baba.terms()[1])[1],
                                                              std::get<2>(temp2baba.terms()[1])[0],
                                                              std::get<2>(temp2baba.terms()[1])[1]};
                                    if ((std::find(uniqe_2bdy.begin(), uniqe_2bdy.end(), vtemp) ==
                                         uniqe_2bdy.end())) {
                                        if ((std::find(adjnt_2bdy.begin(), adjnt_2bdy.end(),
                                                       vtemp) == adjnt_2bdy.end())) {
                                            uniqe_2bdy.push_back(vtemp);
                                            adjnt_2bdy.push_back(vadjt);
                                            add_term(1.0, temp2baba);
                                        }
                                    }
                                }
                            }

                            if ((aa != bb) && (ib != ja)) {
                                SQOperator temp2abba;
                                temp2abba.add_term(+1.0, {aa, bb}, {ib, ja});
                                temp2abba.add_term(-1.0, {ja, ib}, {bb, aa});
                                temp2abba.simplify();
                                if (temp2abba.terms().size() > 0) {
                                    std::vector<size_t> vtemp{std::get<1>(temp2abba.terms()[0])[0],
                                                              std::get<1>(temp2abba.terms()[0])[1],
                                                              std::get<2>(temp2abba.terms()[0])[0],
                                                              std::get<2>(temp2abba.terms()[0])[1]};
                                    std::vector<size_t> vadjt{std::get<1>(temp2abba.terms()[1])[0],
                                                              std::get<1>(temp2abba.terms()[1])[1],
                                                              std::get<2>(temp2abba.terms()[1])[0],
                                                              std::get<2>(temp2abba.terms()[1])[1]};
                                    if ((std::find(uniqe_2bdy.begin(), uniqe_2bdy.end(), vtemp) ==
                                         uniqe_2bdy.end())) {
                                        if ((std::find(adjnt_2bdy.begin(), adjnt_2bdy.end(),
                                                       vtemp) == adjnt_2bdy.end())) {
                                            uniqe_2bdy.push_back(vtemp);
                                            adjnt_2bdy.push_back(vadjt);
                                            add_term(1.0, temp2abba);
                                        }
                                    }
                                }
                            }

                            if ((ab != ba) && (ia != jb)) {
                                SQOperator temp2baab;
                                temp2baab.add_term(+1.0, {ab, ba}, {ia, jb});
                                temp2baab.add_term(-1.0, {jb, ia}, {ba, ab});
                                temp2baab.simplify();
                                if (temp2baab.terms().size() > 0) {
                                    std::vector<size_t> vtemp{std::get<1>(temp2baab.terms()[0])[0],
                                                              std::get<1>(temp2baab.terms()[0])[1],
                                                              std::get<2>(temp2baab.terms()[0])[0],
                                                              std::get<2>(temp2baab.terms()[0])[1]};
                                    std::vector<size_t> vadjt{std::get<1>(temp2baab.terms()[1])[0],
                                                              std::get<1>(temp2baab.terms()[1])[1],
                                                              std::get<2>(temp2baab.terms()[1])[0],
                                                              std::get<2>(temp2baab.terms()[1])[1]};
                                    if ((std::find(uniqe_2bdy.begin(), uniqe_2bdy.end(), vtemp) ==
                                         uniqe_2bdy.end())) {
                                        if ((std::find(adjnt_2bdy.begin(), adjnt_2bdy.end(),
                                                       vtemp) == adjnt_2bdy.end())) {
                                            uniqe_2bdy.push_back(vtemp);
                                            adjnt_2bdy.push_back(vadjt);
                                            add_term(1.0, temp2baab);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } else if ((pool_type == "S") || (pool_type == "SD") || (pool_type == "SDT") ||
               (pool_type == "SDTQ") || (pool_type == "SDTQP") || (pool_type == "SDTQPH")) {

        int max_nbody = 0;

        if (pool_type == "S") {
            max_nbody = 1;
        } else if (pool_type == "SD") {
            max_nbody = 2;
        } else if (pool_type == "SDT") {
            max_nbody = 3;
        } else if (pool_type == "SDTQ") {
            max_nbody = 4;
        } else if (pool_type == "SDTQP") {
            max_nbody = 5;
        } else if (pool_type == "SDTQPH") {
            max_nbody = 6;
        } else {
            throw std::invalid_argument("Qforte UCC only supports up to Hextuple excitations.");
        }

        std::bitset<64> ref_bin(ref_int_);
        auto ref_bin_str = ref_bin.to_string().substr(64 - n_spinorb_);

        std::string alpha_str = "";
        std::string beta_str = "";

        for (size_t i = 0; i < ref_bin_str.length(); i++) {
            if (i % 2 == 0) {
                beta_str += ref_bin_str[i];
            } else {
                alpha_str += ref_bin_str[i];
            }
        }

        std::vector<std::string> alpha_permutations{};
        std::vector<std::string> beta_permutations{};

        // compute all possible excitations of the n_occ_alpha_ electrons
        do
            alpha_permutations.push_back(alpha_str);
        while (std::next_permutation(alpha_str.begin(), alpha_str.end()));

        // compute all possible excitations of the n_occ_beta_ electrons
        do
            beta_permutations.push_back(beta_str);
        while (std::next_permutation(beta_str.begin(), beta_str.end()));

        // construct N- and Sz-symmetry preserving determinants
        std::vector<uint64_t> dets_int{};
        for (const auto& str_alpha : alpha_permutations) {
            for (const auto& str_beta : beta_permutations) {
                std::string det_str = "";
                for (size_t i = 0; i < str_alpha.length(); i++) {
                    det_str += str_beta[i];
                    det_str += str_alpha[i];
                }
                std::bitset<64> det_bit(det_str);
                dets_int.push_back(det_bit.to_ullong());
            }
        }

        // To reproduce the results of older versions of QForte, the dets are sorted
        std::sort(dets_int.begin(), dets_int.end());

        for (const auto& det_int : dets_int) {
            // Create the bitstring of created/annihilated orbitals
            std::bitset<64> excit(ref_int_ ^ det_int);
            auto excit_str = excit.to_string().substr(64 - n_spinorb_);
            int n_excitation_indices = std::count(excit_str.begin(), excit_str.end(), '1');
            if (n_excitation_indices % 2 != 0) {
                throw std::invalid_argument(
                    "The number of excitation indices must be an even number!");
            }
            int excit_rank = n_excitation_indices / 2;
            // Confirm excitation number is non-zero and consider operators with rank <=
            // max_excit_rank
            if (excit_rank != 0 && excit_rank <= max_nbody) {
                // Get the indices of occupied and unoccupied orbitals
                std::vector<size_t> occ_idx;
                std::vector<size_t> unocc_idx;
                for (size_t i = 0; i < excit_str.size(); i++) {
                    if (excit_str[i] != '1') {
                        continue;
                    }
                    if (ref_bin_str[i] == '1') {
                        occ_idx.push_back(n_spinorb_ - i - 1);
                    } else if (ref_bin_str[i] == '0') {
                        unocc_idx.push_back(n_spinorb_ - i - 1);
                    }
                }
                // impose spatial symmetry
                std::vector<size_t> excitation_indices;
                excitation_indices.reserve(occ_idx.size() + unocc_idx.size());
                excitation_indices.insert(excitation_indices.end(), occ_idx.begin(), occ_idx.end());
                excitation_indices.insert(excitation_indices.end(), unocc_idx.begin(),
                                          unocc_idx.end());
                if (!find_irrep(orb_irreps_to_int_, excitation_indices)) {
                    // construct second-quantized operator and add to pool
                    SQOperator t_temp;
                    t_temp.add_term(+1.0, unocc_idx, occ_idx);
                    std::vector<size_t> runocc_idx(unocc_idx.rbegin(), unocc_idx.rend());
                    std::vector<size_t> rocc_idx(occ_idx.rbegin(), occ_idx.rend());
                    t_temp.add_term(-1.0, rocc_idx, runocc_idx);
                    t_temp.simplify();
                    add_term(1.0, t_temp);
                }
            }
        }
    } else if (pool_type == "sa_SD") {
        if (n_occ_alpha_ != n_occ_beta_) {
            throw std::invalid_argument(
                "sa_SD operator pool requires a closed-shell reference determinant!");
        }
        int nocc_ = n_occ_alpha_;
        int nvir_ = n_vir_alpha_;
        for (size_t i = 0; i < nocc_; i++) {
            size_t ia = 2 * i;
            size_t ib = 2 * i + 1;

            for (size_t a = 0; a < nvir_; a++) {
                size_t aa = 2 * nocc_ + 2 * a;
                size_t ab = 2 * nocc_ + 2 * a + 1;

                if (!find_irrep(orb_irreps_to_int_, std::vector<size_t>{ia, aa})) {

                    SQOperator temp1;
                    temp1.add_term(+1.0 / std::sqrt(2), {aa}, {ia});
                    temp1.add_term(+1.0 / std::sqrt(2), {ab}, {ib});

                    temp1.add_term(-1.0 / std::sqrt(2), {ia}, {aa});
                    temp1.add_term(-1.0 / std::sqrt(2), {ib}, {ab});

                    temp1.simplify();

                    add_term(1.0, temp1);
                }
            }
        }

        for (size_t i = 0; i < nocc_; i++) {
            size_t ia = 2 * i;
            size_t ib = 2 * i + 1;

            for (size_t j = i; j < nocc_; j++) {
                size_t ja = 2 * j;
                size_t jb = 2 * j + 1;

                for (size_t a = 0; a < nvir_; a++) {
                    size_t aa = 2 * nocc_ + 2 * a;
                    size_t ab = 2 * nocc_ + 2 * a + 1;

                    for (size_t b = a; b < nvir_; b++) {
                        size_t ba = 2 * nocc_ + 2 * b;
                        size_t bb = 2 * nocc_ + 2 * b + 1;

                        if (!find_irrep(orb_irreps_to_int_, std::vector<size_t>{ia, ja, aa, ba})) {

                            SQOperator temp2a;
                            if ((aa != ba) && (ia != ja)) {
                                temp2a.add_term(2.0 / std::sqrt(12), {aa, ba}, {ia, ja});
                            }
                            if ((ab != bb) && (ib != jb)) {
                                temp2a.add_term(2.0 / std::sqrt(12), {ab, bb}, {ib, jb});
                            }
                            if ((aa != bb) && (ia != jb)) {
                                temp2a.add_term(1.0 / std::sqrt(12), {aa, bb}, {ia, jb});
                            }
                            if ((ab != ba) && (ib != ja)) {
                                temp2a.add_term(1.0 / std::sqrt(12), {ab, ba}, {ib, ja});
                            }
                            if ((aa != bb) && (ib != ja)) {
                                temp2a.add_term(1.0 / std::sqrt(12), {aa, bb}, {ib, ja});
                            }
                            if ((ab != ba) && (ia != jb)) {
                                temp2a.add_term(1.0 / std::sqrt(12), {ab, ba}, {ia, jb});
                            }

                            // hermitian conjugate
                            if ((ja != ia) && (ba != aa)) {
                                temp2a.add_term(-2.0 / std::sqrt(12), {ja, ia}, {ba, aa});
                            }
                            if ((jb != ib) && (bb != ab)) {
                                temp2a.add_term(-2.0 / std::sqrt(12), {jb, ib}, {bb, ab});
                            }
                            if ((jb != ia) && (bb != aa)) {
                                temp2a.add_term(-1.0 / std::sqrt(12), {jb, ia}, {bb, aa});
                            }
                            if ((ja != ib) && (ba != ab)) {
                                temp2a.add_term(-1.0 / std::sqrt(12), {ja, ib}, {ba, ab});
                            }
                            if ((ja != ib) && (bb != aa)) {
                                temp2a.add_term(-1.0 / std::sqrt(12), {ja, ib}, {bb, aa});
                            }
                            if ((jb != ia) && (ba != ab)) {
                                temp2a.add_term(-1.0 / std::sqrt(12), {jb, ia}, {ba, ab});
                            }

                            SQOperator temp2b;
                            if ((aa != bb) && (ia != jb)) {
                                temp2b.add_term(0.5, {aa, bb}, {ia, jb});
                            }
                            if ((ab != ba) && (ib != ja)) {
                                temp2b.add_term(0.5, {ab, ba}, {ib, ja});
                            }
                            if ((aa != bb) && (ib != ja)) {
                                temp2b.add_term(-0.5, {aa, bb}, {ib, ja});
                            }
                            if ((ab != ba) && (ia != jb)) {
                                temp2b.add_term(-0.5, {ab, ba}, {ia, jb});
                            }

                            // hermetian conjugate
                            if ((jb != ia) && (bb != aa)) {
                                temp2b.add_term(-0.5, {jb, ia}, {bb, aa});
                            }
                            if ((ja != ib) && (ba != ab)) {
                                temp2b.add_term(-0.5, {ja, ib}, {ba, ab});
                            }
                            if ((ja != ib) && (bb != aa)) {
                                temp2b.add_term(0.5, {ja, ib}, {bb, aa});
                            }
                            if ((jb != ia) && (ba != ab)) {
                                temp2b.add_term(0.5, {jb, ia}, {ba, ab});
                            }

                            temp2a.simplify();
                            temp2b.simplify();

                            std::complex<double> temp2a_norm(0.0, 0.0);
                            std::complex<double> temp2b_norm(0.0, 0.0);
                            for (const auto& term : temp2a.terms()) {
                                temp2a_norm += std::norm(std::get<0>(term));
                            }
                            for (const auto& term : temp2b.terms()) {
                                temp2b_norm += std::norm(std::get<0>(term));
                            }
                            temp2a.mult_coeffs(std::sqrt(2.0 / temp2a_norm));
                            temp2b.mult_coeffs(std::sqrt(2.0 / temp2b_norm));

                            if (temp2a.terms().size() > 0) {
                                add_term(1.0, temp2a);
                            }
                            if (temp2b.terms().size() > 0) {
                                add_term(1.0, temp2b);
                            }
                        }
                    }
                }
            }
        }
    } else {
        throw std::invalid_argument("Invalid pool_type specified.");
    }
}

std::string SQOpPool::str() const {
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
        s.push_back("]\n\n");
        counter++;
    }
    return join(s, " ");
}
