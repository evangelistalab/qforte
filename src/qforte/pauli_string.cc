#include "pauli_string.h"

std::string PauliString::str() const {
    std::string s;
    s += "[";
    for (int i = 0; i < BitString::max_bits_; ++i) {
        if (X_.get_bit(i) and Z_.get_bit(i)) {
            s += " Y" + std::to_string(i);
        }else{
            if (X_.get_bit(i)) {
                s += " X" + std::to_string(i);
            }        
            if (Z_.get_bit(i)) {
                s += " Z" + std::to_string(i);
            }        
        }
    }   
    s += "]";
    return s;
}

// A equivalence comparison for Circuit class
bool operator==(const PauliString& lhs, const PauliString& rhs){
    return (lhs.X() == rhs.X()) and (lhs.Z() == rhs.Z());
}

std::pair<std::complex<double>,PauliString>
multiply(const PauliString& lhs, const PauliString& rhs){
    std::complex<double> phase = 1.0;
    PauliString ps(lhs.X() ^ rhs.X(),lhs.Z() ^ rhs.Z());
    return std::make_pair(phase,ps);
}