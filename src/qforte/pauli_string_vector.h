#ifndef _pauli_string_vector_h_
#define _pauli_string_vector_h_

#include <algorithm>
#include <complex>
#include <vector>

#include "pauli_string.h"
#include "gate.h"
#include "circuit.h"
#include "qubit_operator.h"

class PauliStringVector {
    /* TODO: fill
     */
  public:
    /// constructors
    PauliStringVector() {}
    PauliStringVector(std::vector<PauliString> PauliVector) : PauliVector_(PauliVector) {}

    /// return a string representing the linear combination of Pauli strings
    std::vector<std::string> str() const;

    /// return the underlying vector
    std::vector<PauliString> get_vec() const;

    /// order the terms by increasing Z and X bitstring values
    void order_terms();

    /// combine idential Pauli strings and eliminate negligible terms
    void simplify();

    /// get number of qubits that PauliStringVector is acting on
    size_t num_qubits() const;

    /// add PauliString object to PauliStringVector object
    void add_PauliString(const PauliString& ps);

    /// add PauliStringVector object
    void add_PauliStringVector(const PauliStringVector& psv);

    /// give PauliStringVector class index access operator
    PauliString& operator[](unsigned int i) {return PauliVector_[i];}
    const PauliString& operator[](unsigned int i) const {return PauliVector_[i];}

    /// convert PauliStringVector to QubitOperator
    QubitOperator get_QubitOperator();

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
