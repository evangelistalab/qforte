#ifndef _qubit_basis_h_
#define _qubit_basis_h_

#include <complex>
#include <numeric>
#include <string>

#include "bitstring.h"
#include "pauli_string.h"
#include "pauli_string_vector.h"

class QubitBasisVector;

/**
 * @brief The QubitBasis class
 *
 * This class represents an element of the Hilbert space basis:
 *   |q_0 q_1 q_2 ... q_(n-1)> with q_i = {0, 1}
 *
 *   for example, with n = 4, some of the elements of this basis include:
 *
 *   |1010>, |0000>, |1110>
 */
class QubitBasis {
  public:
    /// constructor
    QubitBasis(size_t n = 0, std::complex<double> coeff = 1.0) : state_(n), coeff_(coeff) {}

    /// the maximum number of qubits
    static size_t max_qubits() {return BitString::max_bits_;}

    /// return the bitstring representation
    const BitString& get_bits() const {return state_;}

    /// return the coefficient of the state
    const std::complex<double> coeff() const {return coeff_;}

    /// get the value of bit pos
    bool get_bit(size_t pos) const { return state_.get_bit(pos); }

    /// set bit in position 'pos' to the boolean val
    void set_bit(size_t pos, bool val) { state_.set_bit(pos,val); }

    /// flip bin in position 'pos' to opposite boolean invalid_argument
    void flip_bit(size_t pos) { state_.flip_bit(pos); }

    // void set(basis_t state) { state_.set(state); }

    void zero() { state_.zero(); }

    size_t add() const { return state_.address(); }
    size_t address() const { return state_.address(); }

    std::string str(size_t nqubit) const;

  private:
    /// the state
    BitString state_;
    std::complex<double> coeff_;
};

QubitBasis multiply(const std::complex<double> lhs, const QubitBasis& rhs);

QubitBasisVector add(const QubitBasis& lhs, const QubitBasis& rhs);

QubitBasisVector subtract(const QubitBasis& lhs, const QubitBasis& rhs);

QubitBasis apply(const PauliString& pauli, const QubitBasis& ket);
QubitBasisVector apply(const PauliStringVector& PauliVector, const QubitBasis& ket);

#endif // _qubit_basis_h_
