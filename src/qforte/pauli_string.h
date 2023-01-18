#ifndef _pauli_string_h_
#define _pauli_string_h_

#include <complex>
#include <vector>

#include "bitstring.h"

class PauliStringVector;

class PauliString {
    /* TODO: fill
     */
  public:
    /// constructor
    PauliString(BitString X = 0, BitString Z = 0, std::complex<double> coeff = 1.0) : X_(X), Z_(Z), coeff_(coeff) {}

    /// return bitstring components of PauliString
    const BitString& X() const {return X_;}
    const BitString& Z() const {return Z_;}

    /// return coefficient of PauliString
    const std::complex<double> coeff() const {return coeff_;}

    /// return string representing PauliString object
    std::string str() const;

  private:
    /// the bitstrings represnting the X and Z gates and the complex coefficient of the Pauli string
    BitString X_;
    BitString Z_;
    std::complex<double> coeff_;
};

//  equality comparison for PauliStrings
bool operator==(const PauliString& lhs, const PauliString& rhs);

// < operation for PauliStrings; useful for sorting
bool operator<(const PauliString& lhs, const PauliString& rhs);

// multiplication of two PauliString objects
PauliString multiply(const PauliString& lhs, const PauliString& rhs);

// multiplication of a PauliString object with a scalar
PauliString multiply(const PauliString& lhs, const std::complex<double> rhs);

// addition and subtraction of two PauliString objects
PauliStringVector add(const PauliString& lhs, const PauliString& rhs);
PauliStringVector subtract(const PauliString& lhs, const PauliString& rhs);

#endif // _pauli_string_h_
