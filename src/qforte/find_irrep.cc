#include "find_irrep.h"

size_t find_irrep(const std::vector<size_t>& orb_irreps_to_int,
                  const std::vector<size_t>& spinorb_indices) {

    size_t irrep = 0;
    for (size_t index : spinorb_indices) {
        irrep ^= orb_irreps_to_int[index / 2];
    }

    return irrep;
}
