#include "bitwise_operations.h"
#include "helpers.h"
#include "pauli_string.h"

std::string PauliString::str() const {
    std::vector<std::string> s;
    for (int i = 0; i < BitString::max_bits_; ++i) {
        if (X_.get_bit(i) and Z_.get_bit(i)) {
            s.push_back("Y" + std::to_string(i));
        }else{
            if (X_.get_bit(i)) {
                s.push_back("X" + std::to_string(i));
            }
            if (Z_.get_bit(i)) {
                s.push_back("Z" + std::to_string(i));
            }
        }
    }
    return "[" + join(s, " ") + "]";
}

// A equivalence comparison for Circuit class
bool operator==(const PauliString& lhs, const PauliString& rhs){
    return (lhs.X() == rhs.X()) and (lhs.Z() == rhs.Z());
}

std::pair<std::complex<double>,PauliString>
multiply(const PauliString& lhs, const PauliString& rhs){
    constexpr std::complex<double> i_powers[] = {{1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}};
    std::complex<double> phase = 1.0;
    uint8_t imaginary_phase = 0;
    uint8_t minus_one_phase = 0;
    const uint64_t rhsX_lhsZ           =                     ui64_bit_count((rhs.X() & lhs.Z()).get_bits());
    const uint64_t rhsZ_lhsX           =                     ui64_bit_count((rhs.Z() & lhs.X()).get_bits());
    const uint64_t rhsX_lhsX_lhsZ      =           ui64_bit_count((rhs.X() & lhs.X() & lhs.Z()).get_bits());
    const uint64_t rhsX_rhsZ_lhsZ      =           ui64_bit_count((rhs.X() & rhs.Z() & lhs.Z()).get_bits());
    const uint64_t rhsX_rhsZ_lhsX      =           ui64_bit_count((rhs.X() & rhs.Z() & lhs.X()).get_bits());
    const uint64_t rhsZ_lhsX_lhsZ      =           ui64_bit_count((rhs.Z() & lhs.X() & lhs.Z()).get_bits());
    const uint64_t rhsX_rhsZ_lhsX_lhsZ = ui64_bit_count((rhs.X() & rhs.Z() & lhs.X() & lhs.Z()).get_bits());
    PauliString ps(lhs.X() ^ rhs.X(),lhs.Z() ^ rhs.Z());
    imaginary_phase += (        rhsX_lhsZ
                        +       rhsZ_lhsX
                        - 2.0 * rhsX_rhsZ_lhsX_lhsZ);
    minus_one_phase += (        rhsX_lhsZ
                        -       rhsX_lhsX_lhsZ
                        -       rhsX_rhsZ_lhsZ
                        +       rhsX_rhsZ_lhsX
                        +       rhsZ_lhsX_lhsZ
                        -       rhsX_rhsZ_lhsX_lhsZ);
    phase *= (1.0 - 2.0*(minus_one_phase & 1)) * i_powers[imaginary_phase & 3];
    return std::make_pair(phase,ps);
}
