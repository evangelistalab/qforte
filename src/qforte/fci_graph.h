#ifndef fci_graph_h
#define fci_graph_h

#include <vector>
#include <unordered_map>
#include <tuple>

struct PairHash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

using Spinmap = std::unordered_map<std::pair<int, int>, std::vector<std::tuple<int, int, int>>, PairHash>;

class FciGraph {
public:

    /// Constructor
    FciGraph(int nalpha, int nbeta, int norb);

    /// Combinutorics helper funciton for binomial coefficients
    int binom(int n, int k);

    /// Build alpha/beta bitstrings to index the FCI Computer
    std::pair<std::vector<uint64_t>, std::unordered_map<uint64_t, size_t>> build_strings(
        int nele, 
        size_t length); 

    /// Construct the FCI Mapping
    // std::unordered_map<std::string, int> build_mapping(
    //     const std::vector<std::string>& str, 
    //     int n, 
    //     const std::unordered_map<int, int>& ind);

    /// Construct the FCI Mapping
    Spinmap build_mapping(
        const std::vector<uint64_t>& strings, 
        int nele, 
        const std::unordered_map<uint64_t, size_t>& index);

    /// Convert to de-excitaiton vector
    // std::vector<std::vector<int>> map_to_deexc(
    //     const std::unordered_map<std::string, int>& mapping, 
    //     int len, 
    //     int norb, 
    //     int n);

    std::vector<std::vector<std::vector<int>>> map_to_deexc(
        const Spinmap& mappings, 
        int states, 
        int norbs,
        int nele);

    std::vector<uint64_t> lexicographic_bitstring_generator(int nele, int norb);

    uint64_t build_string_address(
        int nele, 
        int norb, 
        uint64_t occ,
        const std::vector<std::vector<uint64_t>>& zmat); 

    std::vector<std::vector<uint64_t>> get_z_matrix(int norb, int nele);

    /// ==> Utility Functions for Bit Math (may need to move) <== ///
    
    bool get_bit(uint64_t string, size_t pos) { return string & maskbit(pos); }

    constexpr uint64_t maskbit(size_t pos) { return static_cast<uint64_t>(1) << pos; }

    int count_bits_between(uint64_t string, size_t pos1, size_t pos2) {
        uint64_t mask = (((1ULL << pos1) - 1) ^ ((1ULL << (pos2 + 1)) - 1)) &
                        (((1ULL << pos2) - 1) ^ ((1ULL << (pos1 + 1)) - 1));
        
        return static_cast<int>(string & mask);
    }

    uint64_t set_bit(uint64_t string, int pos) {
        return string | (1ULL << pos);
    }

    uint64_t unset_bit(uint64_t string, int pos) {
        return string & ~(1ULL << pos);
    }

    /// ==> Setters and Getters <== /// 

    /// return the number of electrons
    size_t get_nalfa() const { return nalpha_; }

private:
    int norb_;
    int nalpha_;
    int nbeta_;
    int lena_;
    int lenb_;

    std::vector<uint64_t> astr_;
    std::vector<uint64_t> bstr_;

    std::unordered_map<uint64_t, size_t> aind_;
    std::unordered_map<uint64_t, size_t> bind_;

    Spinmap alpha_map_;
    Spinmap beta_map_;

    std::vector<std::vector<std::vector<int>>> dexca_;
    std::vector<std::vector<std::vector<int>>> dexcb_;

    
};

#endif
