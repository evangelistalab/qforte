#include "fmt/format.h"

#include "bitwise_operations.h"
#include "helpers.h"
#include "pauli_string.h"
#include "pauli_string_vector.h"

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
    return fmt::format("({:+f} {:+f} i) {}", std::real(coeff_), std::imag(coeff_), "[" + join(s, " ") + "]");
}

bool operator==(const PauliString& lhs, const PauliString& rhs){
    return (lhs.X() == rhs.X()) and (lhs.Z() == rhs.Z()) and (lhs.coeff() == rhs.coeff());
}

bool operator<(const PauliString& lhs, const PauliString& rhs){
    if (ui64_bit_count((lhs.X() | lhs.Z()).get_bits()) < ui64_bit_count((rhs.X() | rhs.Z()).get_bits())) return true;
    if (ui64_bit_count((lhs.X() | lhs.Z()).get_bits()) > ui64_bit_count((rhs.X() | rhs.Z()).get_bits())) return false;
    if (lhs.Z() < rhs.Z()) return true;
    if (lhs.Z() > rhs.Z()) return false;
    if (lhs.X() < rhs.X()) return true;
    return false;
}


PauliString multiply(const PauliString& lhs, const PauliString& rhs){
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
    PauliString ps(lhs.X() ^ rhs.X(),lhs.Z() ^ rhs.Z(), lhs.coeff() * rhs.coeff() * phase);
    return ps;
}

PauliString multiply(const PauliString& lhs, const std::complex<double> rhs){
    PauliString ps(lhs.X(), lhs.Z(), lhs.coeff() * rhs);
    return ps;
}

PauliStringVector add(const PauliString& lhs, const PauliString& rhs){
    std::vector<PauliString> vec{lhs, rhs};
    PauliStringVector PauliVector(vec);
    return PauliVector;
}

PauliStringVector subtract(const PauliString& lhs, const PauliString& rhs){
    std::vector<PauliString> vec{lhs};
    PauliString rhs_minus(rhs.X(), rhs.Z(), -rhs.coeff());
    vec.push_back(rhs_minus);
    PauliStringVector PauliVector(vec);
    return PauliVector;
}
