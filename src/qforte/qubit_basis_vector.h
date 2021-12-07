#ifndef _qubit_basis_vector_h_
#define _qubit_basis_vector_h_

#include <complex>
#include <vector>

#include "qubit_basis.h"

class QubitBasisVector {
    /* TODO: fill
     */
  public:
    /// default constructor: creates a qubit basis vector
    QubitBasisVector(std::vector<QubitBasis> QBasisVector) : QBasisVector_(QBasisVector) {}

    /// return a string representing the linear combination of qubit basis states
    std::vector<std::string> str() const;

    /// return the underlying vector
    std::vector<QubitBasis> get_vec() const;

 //   /// return the number of QubitBasis elements in a QubitBasisVector
 //   unsigned int size() const;

 //   /// return begin and end iterators
 //   unsigned int begin() const;
 //   unsigned int end() const;

    /// give QubitBasisVector class index access operator
    QubitBasis& operator[](unsigned int i) {return QBasisVector_[i];}
    const QubitBasis& operator[](unsigned int i) const {return QBasisVector_[i];}

  private:
    std::vector<QubitBasis> QBasisVector_;
};

QubitBasisVector multiply(const QubitBasisVector& lhs, const std::complex<double> rhs);

QubitBasisVector add(const QubitBasisVector& lhs, const QubitBasis& rhs);
QubitBasisVector add(const QubitBasisVector& lhs, const QubitBasisVector& rhs);

QubitBasisVector subtract(const QubitBasisVector& lhs, const QubitBasis& rhs);
QubitBasisVector subtract(const QubitBasis& lhs, const QubitBasisVector& rhs);
QubitBasisVector subtract(const QubitBasisVector& lhs, const QubitBasisVector& rhs);

QubitBasisVector apply(const QubitBasisVector& lhs, const PauliString& rhs);
QubitBasisVector apply(const QubitBasisVector& lhs, const PauliStringVector& rhs);

#endif // _qubit_basis_vector_h_
