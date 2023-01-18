#ifndef _qubit_basis_h_
#define _qubit_basis_h_

#include <numeric>
#include <string>

#include "bitstring.h"

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
    QubitBasis(size_t n = 0) : state_(n) {}

    /// the maximum number of qubits
    static size_t max_qubits() {return BitString::max_bits_;}

    /// get the value of bit pos
    bool get_bit(size_t pos) const { return state_.get_bit(pos); }

    /// set bit in position 'pos' to the boolean val
    void set_bit(size_t pos, bool val) { state_.set_bit(pos,val); }

    /// flip bin in position 'pos' to opposite boolean invalid_argument
    void flip_bit(size_t pos) { state_.flip_bit(pos); }

    void zero() { state_.zero(); }

    size_t add() const { return state_.address(); }
    size_t address() const { return state_.address(); }

    std::string str(size_t nqubit) const;

  private:
    /// the state
    BitString state_;
};

#endif // _qubit_basis_h_
