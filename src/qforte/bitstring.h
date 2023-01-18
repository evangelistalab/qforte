#ifndef _bitstring_h_
#define _bitstring_h_

#include <numeric>
#include <string>

/**
 * @brief The BitString class
 *
 * This class represents an array of bits:
 * 
 *   [q_0,q_1,q_2,...,q_(n-1)] with q_i = {0, 1}
 *
 */
class BitString {
  public:
    /// the type used to represent a bit array (a 64 bit unsigned long)
    using bit_t = uint64_t;

    /// the maximum number of qubits
    static constexpr size_t max_bits_ = 8 * sizeof(bit_t);

    /// constructor
    BitString(size_t n = 0) { bits_ = n; }

    /// a mask for bit in position pos
    /// for example if pos = 2, maskbit = 100000 << 2 = 001000...
    static bit_t maskbit(size_t pos) { return static_cast<bit_t>(1) << pos; }

    const bit_t& get_bits() const { return bits_;}

    /// get the value of bit pos
    bool get_bit(size_t pos) const { return bits_ & maskbit(pos); }

    /// set bit in position 'pos' to the boolean val
    /// val = 0 -> -val = 0000000...
    /// val = 1 -> -val = 1111111...
    void set_bit(size_t pos, bool val) { bits_ ^= (-val ^ bits_) & maskbit(pos); }

    /// flip bin in position 'pos' to opposite boolean invalid_argument
    void flip_bit(size_t pos) { bits_ ^= maskbit(pos); }

    /// set the bits to a given value
    void set(bit_t bits) { bits_ = bits; };

    /// zero all the bits
    void zero() { bits_ = static_cast<bit_t>(0); }

    /// return the address of this bit array
    size_t address() const { return bits_; }

    /// convert to a string
    std::string str(size_t nbit) const;

  private:
    /// the state
    bit_t bits_;
};

bool operator==(const BitString& lhs, const BitString& rhs);
bool operator<(const BitString& lhs, const BitString& rhs);
bool operator>(const BitString& lhs, const BitString& rhs);
BitString operator^(const BitString& lhs, const BitString& rhs);
BitString operator&(const BitString& lhs, const BitString& rhs);
BitString operator|(const BitString& lhs, const BitString& rhs);

#endif // _bitstring_h_
