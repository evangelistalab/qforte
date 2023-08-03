#include "fci_graph.h"
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <unordered_map>


FciGraph::FciGraph(int nalpha, int nbeta, int norb) {
    if (norb < 0)
        throw std::invalid_argument("norb needs to be >= 0");
    if (nalpha < 0)
        throw std::invalid_argument("nalpha needs to be >= 0");
    if (nbeta < 0)
        throw std::invalid_argument("nbeta needs to be >= 0");
    if (nalpha > norb)
        throw std::invalid_argument("nalpha needs to be <= norb");
    if (nbeta > norb)
        throw std::invalid_argument("nbeta needs to be <= norb");

    norb_ = norb;
    nalpha_ = nalpha;
    nbeta_ = nbeta;
    lena_ = binom(norb, nalpha);  // size of alpha-Hilbert space
    lenb_ = binom(norb, nbeta);   // size of beta-Hilbert space

    // _astr = {};  // string labels for alpha-Hilbert space
    // _bstr = {};  // string labels for beta-Hilbert space
    // _aind = {};  // map string-binary to matrix index
    // _bind = {};  // map string-binary to matrix index

    std::tie(astr_, aind_) = build_strings(nalpha_, lena_);
    std::tie(bstr_, bind_) = build_strings(nbeta_, lenb_);

    alpha_map_ = build_mapping(astr_, nalpha_, aind_);
    beta_map_ = build_mapping(bstr_, nbeta_, bind_);

    dexca_ = map_to_deexc(alpha_map_, lena_, norb_, nalpha_);
    dexcb_ = map_to_deexc(beta_map_, lenb_, norb_, nbeta_);

    // _fci_map = {};
}

int FciGraph::binom(int n, int k) {
    if (k > n - k)
        k = n - k;
    int res = 1;
    for (int i = 0; i < k; ++i) {
        res *= (n - i);
        res /= (i + 1);
    }
    return res;
}

std::pair<std::vector<uint64_t>, std::unordered_map<uint64_t, size_t>> FciGraph::build_strings(
    int nele, 
    size_t length) 
{
    int norb = norb_;

    /// NICK: rename...
    std::vector<uint64_t> blist = lexicographic_bitstring_generator(nele, norb); // Assuming lexicographic_bitstring_generator is available

    std::vector<uint64_t> string_list;

    std::unordered_map<uint64_t, size_t> index_list;

    std::vector<std::vector<uint64_t>> Z = get_z_matrix(norb, nele);

    string_list = std::vector<uint64_t>(length, 0);

    for (size_t i = 0; i < length; ++i) {

        uint64_t occ = blist[i];

        size_t address = build_string_address(
            nele, 
            norb, 
            occ,
            Z); 

        string_list[address] = occ;
    }

    for (size_t address = 0; address < string_list.size(); ++address) {
        uint64_t wbit = string_list[address];
        index_list[wbit] = address;
    }

    return std::make_pair(string_list, index_list);
}

// struct PairHash {
//     template <class T1, class T2>
//     std::size_t operator () (const std::pair<T1, T2>& p) const {
//         auto h1 = std::hash<T1>{}(p.first);
//         auto h2 = std::hash<T2>{}(p.second);
//         return h1 ^ (h2 << 1);
//     }
// };

// using Spinmap = std::unordered_map<std::pair<int, int>, std::vector<std::tuple<int, int, int>>, PairHash>;

Spinmap FciGraph::build_mapping(
    const std::vector<uint64_t>& strings, 
    int nele, 
    const std::unordered_map<uint64_t, size_t>& index) 
{
    int norb = norb_;
    Spinmap out;

    for (int iorb = 0; iorb < norb; ++iorb) {
        for (int jorb = 0; jorb < norb; ++jorb) {
            std::vector<std::tuple<int, int, int>> value;
            for (uint64_t string : strings) {
                if (get_bit(string, jorb) && !get_bit(string, iorb)) {
                    int parity = count_bits_between(string, iorb, jorb); // Assuming count_bits_between is available
                    int sign = (parity % 2 == 0) ? 1 : -1;
                    value.push_back(std::make_tuple(index.at(string), index.at(unset_bit(set_bit(string, iorb), jorb)), sign));
                } else if (iorb == jorb && get_bit(string, iorb)) {
                    value.push_back(std::make_tuple(index.at(string), index.at(string), 1));
                }
            }
            out[std::make_pair(iorb, jorb)] = value;
        }
    }

    Spinmap result;
    for (const auto& entry : out) {
        const auto& key = entry.first;
        const auto& value = entry.second;
        // std::vector<std::vector<int>> casted_value;
        std::vector<std::tuple<int,int,int>> casted_value;
        for (const auto& tpl : value) {
            casted_value.push_back(std::make_tuple(std::get<0>(tpl), std::get<1>(tpl), std::get<2>(tpl)));
        }
        result[key] = casted_value;
    }

    return result;
}

// bool get_bit(uint64_t string, size_t pos) { return string & maskbit(pos); }

// constexpr uint64_t maskbit(size_t pos) { return static_cast<uint64_t>(1) << pos; }

// int count_bits_between(uint64_t string, size_t pos1, size_t pos2) {
//     uint64_t mask = (((1ULL << pos1) - 1) ^ ((1ULL << (pos2 + 1)) - 1)) &
//                     (((1ULL << pos2) - 1) ^ ((1ULL << (pos1 + 1)) - 1));
    
//     return static_cast<int>(string & mask);
// }

// uint64_t set_bit(uint64_t string, int pos) {
//     return string | (1ULL << pos);
// }

// uint64_t unset_bit(uint64_t string, int pos) {
//     return string & ~(1ULL << pos);
// }

std::vector<std::vector<std::vector<int>>> FciGraph::map_to_deexc(
    const Spinmap& mappings, 
    int states, 
    int norbs,
    int nele) 
{
    int lk = nele * (norbs - nele + 1);
    std::vector<std::vector<std::vector<int>>> dexc(
        states, 
        std::vector<std::vector<int>>(lk, std::vector<int>(3, 0)));

    std::vector<int> index(states, 0);
    
    for (const auto& entry : mappings) {
        const auto& key = entry.first;
        const auto& values = entry.second;
        int i = key.first;
        int j = key.second;
        int idx = i * norbs + j;
        
        for (const auto& value : values) {
            int state = std::get<0>(value);
            int target = std::get<1>(value);
            int parity = std::get<2>(value);
            
            dexc[target][index[target]][0] = state;
            dexc[target][index[target]][1] = idx;
            dexc[target][index[target]][2] = parity;
            index[target]++;
        }
    }
    
    return dexc;
}

// Additional helper functions
// uint64_t FciGraph::integer_index(uint64_t wbit) {

// }

/// NICK: 1. Consider a faster blas veriosn, 2. consider using qubit basis, 3. rename (too long)
std::vector<uint64_t> FciGraph::lexicographic_bitstring_generator(int nele, int norb) {

    if (nele > norb) {
        throw std::invalid_argument("can't have more electorns that orbitals");
    }
        
    std::vector<uint64_t> bitstrings;

    std::vector<uint64_t> indices(norb);
    for (int i = 0; i < norb; ++i)
        indices[i] = i;

    std::vector<bool> bitstring(norb, false);
    for (int i = 0; i < nele; ++i)
        bitstring[i] = true;

    do {
        uint64_t state = 0;
        for (int i = 0; i < norb; ++i) {
            if (bitstring[i]) { state |= (static_cast<uint64_t>(1) << i);}
        }
        bitstrings.push_back(state);
    } while (std::prev_permutation(bitstring.begin(), bitstring.end()));

    std::sort(bitstrings.begin(), bitstrings.end());

    return bitstrings;

}


/// NICK: Seems slow..., may want to use qubit basis, convert to size_t maybe??
uint64_t FciGraph::build_string_address(
    int nele, 
    int norb, 
    uint64_t occ,
    const std::vector<std::vector<uint64_t>>& zmat) 
{

    std::vector<int> occupations;

    for (int i = 0; i < 64; ++i) { // Assuming uint64_t is 64 bits
        if (occ & (1ULL << i)) { occupations.push_back(i); }
    }

    uint64_t address = 0;
    for (int i = 0; i < nele; ++i) {
        address += zmat[i][occupations[i]];
    }

    return address;
}

// std::unordered_map<uint64_t, int> FciGraph::build_string_address(int nele, int norb, uint64_t occ) {

// }

// std::vector<uint64_t> FciGraph::calculate_string_address(const std::vector<std::vector<int>>& Z, int nele, int norb, const std::vector<uint64_t>& blist){

// }

/// NICK: May want to make faster using blas calls if it becomes a bottleneck
std::vector<std::vector<uint64_t>> FciGraph::get_z_matrix(int norb, int nele) {
    std::vector<std::vector<uint64_t>> Z(nele, std::vector<uint64_t>(norb, 0)); // Initialize Z matrix with zeros

    if (nele == 0 || norb == 0) {
        return Z; // Return an empty matrix if nele or norb is zero
    }

    for (int k = 1; k < nele; ++k) {
        for (int ll = k; ll < norb - nele + k + 1; ++ll) {
            Z[k - 1][ll - 1] = 0;
            for (int m = norb - ll + 1; m < norb - k + 1; ++m) {
                Z[k - 1][ll - 1] += binom(m, nele - k) - binom(m - 1, nele - k - 1);
            }
        }
    }

    int k = nele;
    for (int ll = nele; ll < norb + 1; ++ll) {
        Z[k - 1][ll - 1] = static_cast<uint64_t>(ll - nele);
    }

    return Z;
}

int binom(int n, int m) {
    if (m < 0 || m > n)
        return 0;

    // Initialize a 2D vector to store the binomial coefficients
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1, 0));

    // Base cases
    for (int i = 0; i <= n; ++i)
        dp[i][0] = 1;

    // Calculate the binomial coefficients using dynamic programming
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= std::min(i, m); ++j) {
            dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
        }
    }

    return dp[n][m];
}
