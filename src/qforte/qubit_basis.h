#include <numeric>
#include <string>

/**
 * @brief The QubitBasis class
 *
 * This class represents an element of the Hilbert space basis:
 *   |q_1 q_2 ... q_n> with q_i = {0, 1}
 *
 *   for example:
 *
 *   |1010>, |0000>, |1110>
 */
class QubitBasis {
  public:
    /// the type used to represent a quantum state (a 64 bit unsigned long)
    using basis_t = uint64_t;

    /// the maximum number of qubits
    static constexpr size_t max_qubits_ = 8 * sizeof(basis_t);

    /// constructor
    QubitBasis(size_t n = static_cast<basis_t>(0)) { state_ = n; }

    /// a mask for bit in position pos
    static constexpr basis_t maskbit(size_t pos) { return static_cast<basis_t>(1) << pos; }

    /// get the value of bit pos
    bool get_bit(size_t pos) const { return state_ & maskbit(pos); }

    /// set bit in position 'pos' to the boolean val
    void set_bit(size_t pos, bool val) { state_ ^= (-val ^ state_) & maskbit(pos); }

    /// flip bin in position 'pos' to opposite boolean invalid_argument
    void flip_bit(size_t pos) { state_ ^= maskbit(pos); }

    /// set this state to a give value
    void set(basis_t state);

    /// set this state to zero
    void zero() { state_ = static_cast<basis_t>(0); }

    /// Convenience function to return state_ as an index.
    size_t index() const { return static_cast<size_t>(state_); }

    /// return the state
    basis_t get_state() const { return state_; }

    /// return a string representing the state showing up to nqubit qubits
    std::string str(size_t nqubit) const;

    /// return a string representing the state showing the maximum number of qubits
    std::string default_str() const { return str(max_qubits_); }

  private:
    /// the state
    basis_t state_;
};

/// equality operator
inline bool operator==(const QubitBasis& lhs, const QubitBasis& rhs) {
    return lhs.get_state() == rhs.get_state();
}
