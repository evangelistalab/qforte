#include "quantum_basis.h"

std::string QuantumBasis::str(size_t nqubit) const {
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

void QuantumBasis::set(basis_t state) { state_ = state; }
