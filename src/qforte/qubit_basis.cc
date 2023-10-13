#include "qubit_basis.h"

std::string QubitBasis::str(size_t nqubit) const {
    std::string s;
    s += "|";
    for (int i = 0; i < nqubit; ++i) {
        if (get_bit(i)) {
            s += "1";
        } else {
            s += "0";
        }
    }
    s += ">";
    return s;
}

size_t QubitBasis::get_num_ones() const {
    size_t num_ones = 0;
    for(size_t pos = 0; pos < 64; pos++) {
        num_ones += get_bit(pos);
    }
    return num_ones;
}

void QubitBasis::set(basis_t state) { state_ = state; }
