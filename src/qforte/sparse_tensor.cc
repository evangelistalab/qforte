#include <algorithm>
#include "sparse_tensor.h"

//// SparseVector
std::complex<double> SparseVector::get_element(size_t I) const {
    std::map<size_t, std::complex<double>>::const_iterator i = values_.find(I);
    if (i == values_.end()) {
        return 0.0;
    }
    return i->second;
}

std::map<size_t, std::complex<double>>::const_iterator
SparseVector::get_element_it(size_t I) const {
    std::map<size_t, std::complex<double>>::const_iterator i = values_.find(I);
    return i;
}

std::map<size_t, std::complex<double>>::const_iterator SparseVector::get_end() const {
    return values_.end();
}

void SparseVector::set_element(size_t I, std::complex<double> val) { values_[I] = val; }

std::map<size_t, std::complex<double>> SparseVector::to_map() const { return values_; }

void SparseVector::erase(size_t I) { values_.erase(I); }

//// SparseMatrix
std::complex<double> SparseMatrix::get_element(size_t I, size_t J) const {
    std::map<size_t, SparseVector>::const_iterator i = values_.find(I);
    if (i == values_.end()) {
        return 0.0;
    }
    return i->second.get_element(J);
}

void SparseMatrix::set_element(size_t I, size_t J, std::complex<double> val) {
    values_[I].set_element(J, val);
}

std::map<size_t, SparseVector> SparseMatrix::to_vec_map() const { return values_; }

std::map<size_t, std::map<size_t, std::complex<double>>> SparseMatrix::to_map() const {
    std::map<size_t, std::map<size_t, std::complex<double>>> sp_mat_map;
    for (auto const& x : values_) {
        sp_mat_map.insert(std::make_pair(x.first, x.second.to_map()));
    }
    return sp_mat_map;
}

std::vector<size_t> SparseMatrix::get_unique_js() const {
    std::vector<size_t> unique_js;
    for (auto const& i : values_) { // i-> [ size_t, SparseVector ]
        for (auto const& j : i.second.to_map()) {
            auto iter = std::find(unique_js.begin(), unique_js.end(), j.first);
            if (iter == unique_js.end()) {
                unique_js.push_back(j.first);
            }
        }
    }
    return unique_js;
}

void SparseMatrix::left_multiply(const SparseMatrix& Lmat) {
    // SparseMatrix temp = SparseMatrix();
    std::map<size_t, SparseVector> temp;

    std::complex<double> val_0_0 = 0.0;

    std::vector<size_t> unique_js = get_unique_js();

    for (auto const& i : Lmat.to_vec_map()) {
        // i -> [ I (for Lmat), SparseVector ]
        for (auto const& j_val : unique_js) {
            std::complex<double> val = 0.0;
            // k -> [ J (for Lmat), complex ]
            for (auto const& k : i.second.to_map()) {
                val += k.second * get_element(k.first, j_val);
            }
            if (std::abs(val) > 1.0e-16) {
                temp[i.first].set_element(j_val, val);
            }
        }
    }
    values_ = temp;
}

void SparseMatrix::add(const SparseMatrix& Mat, const std::complex<double> factor) {

    std::map<size_t, SparseVector> temp = values_;

    for (auto const& i : Mat.to_vec_map()) {      // i-> [ size_t, SparseVector ]
        for (auto const& j : i.second.to_map()) { // j-> [ size_t, complex double ]
            std::complex<double> val = factor * j.second + get_element(i.first, j.first);
            if (std::abs(val) > 1.0e-16) {
                temp[i.first].set_element(j.first, val);
            } else {
                // make sure to remove from map, otherwise it will remain the old Value
                temp[i.first].erase(j.first);
            }
        }
    }
    values_ = temp;
}

void SparseMatrix::make_identity(const size_t nbasis) {
    std::map<size_t, SparseVector> temp;
    for (size_t I = 0; I < nbasis; I++) {
        temp[I].set_element(I, 1.0);
    }
    values_ = temp;
}

// end comment
