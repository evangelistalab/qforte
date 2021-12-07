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
    /// default constructor: creates an empty Pauli string
    PauliString(BitString X = 0, BitString Z = 0, std::complex<double> coeff = 1.0) : X_(X), Z_(Z), coeff_(coeff) {}

    // void add_gate(size_t qubit, const std::string& type);

    /// return a string representing this quantum operator
    const BitString& X() const {return X_;}
    const BitString& Z() const {return Z_;}
    const std::complex<double> coeff() const {return coeff_;}

    /// return a string representing this quantum operator
    std::string str() const;

  private:
    /// the linear combination of circuits
    BitString X_;
    BitString Z_;
    std::complex<double> coeff_;
};

// A equivalence comparison for Circuit class
bool operator==(const PauliString& lhs, const PauliString& rhs);
PauliString multiply(const PauliString& lhs, const PauliString& rhs);
PauliString multiply(const PauliString& lhs, const std::complex<double> rhs);

PauliStringVector add(const PauliString& lhs, const PauliString& rhs);

PauliStringVector subtract(const PauliString& lhs, const PauliString& rhs);

#endif // _pauli_string_h_
