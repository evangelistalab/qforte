#include "bitstring.h"

std::string BitString::str(size_t nbit) const {
    std::string s;
    s += "|";
    for (int i = 0; i < nbit; ++i) {
        if (get_bit(i)) {
            s += "1";
        } else {
            s += "0";
        }
    }
    s += ">";
    return s;
}

bool operator==(const BitString& lhs, const BitString& rhs){
    return lhs.get_bits() == rhs.get_bits();
}

BitString operator^(const BitString& lhs, const BitString& rhs)
{
    return BitString(lhs.get_bits() ^ rhs.get_bits());
}

BitString operator&(const BitString& lhs, const BitString& rhs)
{
    return BitString(lhs.get_bits() & rhs.get_bits());
}