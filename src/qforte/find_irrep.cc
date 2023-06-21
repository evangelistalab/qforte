#include "find_irrep.h"

int find_irrep(const std::vector<int>& orb_irreps_to_int,
        const std::vector<int>& spinorb_indices) {

    int irrep = 0;
    for (int index : spinorb_indices) {
        irrep ^= orb_irreps_to_int[index/2];
    }

    return irrep;
}   
