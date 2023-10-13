#include <numeric>
#include <string>
#include <cstdint>

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

    void set(basis_t state);

    void zero() { state_ = static_cast<basis_t>(0); }

    // TODO: Rename to get_state.
    size_t add() const { return state_; }

    std::string str(size_t nqubit) const;

    std::string default_str() const { return str(max_qubits_); }

  private:
    /// the state
    basis_t state_;
};
