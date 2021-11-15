#ifndef _bitwise_operations_h_
#define _bitwise_operations_h_

#ifdef __SSE4_2__
#include <nmmintrin.h>
#endif

#define USE_builtin_popcountll 1

#include <cstdint>

/**
 * @brief Count the number of bits set to 1 in a uint64_t
 * @param x the uint64_t integer to test
 * @return the number of bits that are set to 1
 *
 * If available, this function uses SSE4.2 instructions (_mm_popcnt_u64) to speed up the evaluation.
 *
 * Adapted from the corresponding function in Forte
 */
inline uint64_t ui64_bit_count(uint64_t x) {
#ifdef __SSE4_2__
    // this version is 2.6 times faster than the one below
    return _mm_popcnt_u64(x);
#else
    x -= (x >> 1) & 0x5555555555555555UL;
    x  = (x & 0x3333333333333333UL) + ((x >> 2) & 0x3333333333333333UL);
    x  = (x + ( x >> 4)) & 0x0f0f0f0f0f0f0f0fUL;
    x +=  x >>  8;
    x +=  x >> 16;
    x +=  x >> 32;
    return x & 0x7f;
#endif
}

#endif // _bitwise_operations_h_
