#include "fmt/format.h"

#include "bitwise_operations.h"
#include "qubit_basis.h"
#include "qubit_basis_vector.h"

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
    return fmt::format("({:+f} {:+f} i) {}", std::real(coeff_), std::imag(coeff_), s);
}

QubitBasis
multiply(const std::complex<double> lhs, const QubitBasis& rhs){
    QubitBasis qb( rhs.get_bits().get_bits(), rhs.coeff() * lhs);
    return qb;
}

QubitBasisVector add(const QubitBasis& lhs, const QubitBasis& rhs){
    std::vector<QubitBasis> kets{lhs, rhs};
    QubitBasisVector QBasisVector(kets);
    return QBasisVector;
}

QubitBasisVector subtract(const QubitBasis& lhs, const QubitBasis& rhs){
    std::vector<QubitBasis> kets{lhs};
    QubitBasis rhs_minus(rhs.get_bits().get_bits(), -rhs.coeff());
    kets.push_back(rhs_minus);
    QubitBasisVector QBasisVector(kets);
    return QBasisVector;
}

QubitBasis apply(const PauliString& pauli, const QubitBasis& ket){
    constexpr std::complex<double> i_powers[] = {{1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}};
    std::complex<double> phase = 1.0;
    const uint8_t imaginary_phase = ui64_bit_count((pauli.X() & pauli.Z()).get_bits());
    const uint8_t minus_one_phase = ui64_bit_count((pauli.Z() & ket.get_bits()).get_bits());
    phase *= (1.0 - 2.0*(minus_one_phase & 1)) * i_powers[imaginary_phase & 3];
    return QubitBasis((pauli.X() ^ ket.get_bits()).get_bits(), pauli.coeff() * ket.coeff() * phase);
}

QubitBasisVector apply(const PauliStringVector& lhs, const QubitBasis& ket){
    std::vector<QubitBasis> kets;
    for(const PauliString& ps: lhs.get_vec()){
        QubitBasis result = apply(ps, ket);
        kets.push_back(result);
    }
    QubitBasisVector QBasisVector(kets);
    return QBasisVector;
}
