#ifndef _pauli_string_vector_h_
#define _pauli_string_vector_h_

#include <complex>
#include <vector>

#include "pauli_string.h"

class PauliStringVector {
    /* TODO: fill
     */
  public:
    /// default constructor: creates a Pauli string vector
    PauliStringVector(std::vector<PauliString> PauliVector) : PauliVector_(PauliVector) {}

    /// return a string representing the linear combination of Pauli strings
    std::vector<std::string> str() const;

    /// return the underlying vector
    std::vector<PauliString> get_vec() const;

 //   /// return the number of PauliString elements in a PauliStringVector
 //   unsigned int size() const;

 //   /// return begin and end iterators
 //   unsigned int begin() const;
 //   unsigned int end() const;

    /// give PauliStringVector class index access operator
    PauliString& operator[](unsigned int i) {return PauliVector_[i];}
    const PauliString& operator[](unsigned int i) const {return PauliVector_[i];}

  private:
    std::vector<PauliString> PauliVector_;
};

PauliStringVector multiply(const PauliStringVector& lhs, const std::complex<double> rhs);
PauliStringVector multiply(const PauliStringVector& lhs, const PauliString& rhs);
PauliStringVector multiply(const PauliString& lhs, const PauliStringVector& rhs);
PauliStringVector multiply(const PauliStringVector& lhs, const PauliStringVector& rhs);

PauliStringVector add(const PauliStringVector& lhs, const PauliString& rhs);
PauliStringVector add(const PauliStringVector& lhs, const PauliStringVector& rhs);

PauliStringVector subtract(const PauliStringVector& lhs, const PauliString& rhs);
PauliStringVector subtract(const PauliString& lhs, const PauliStringVector& rhs);
PauliStringVector subtract(const PauliStringVector& lhs, const PauliStringVector& rhs);


#endif // _pauli_string_vector_h_
