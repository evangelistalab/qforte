#ifndef _fci_graph_h_
#define _fci_graph_h_

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

class FCIGraph {
public:

    /// Constructor
    FCIGraph(int nalfa, int nbeta, int norb);

    FCIGraph();

    /// Build alfa/beta bitstrings to index the FCI Computer
    std::pair<std::vector<uint64_t>, std::unordered_map<uint64_t, size_t>> build_strings(
        int nele, 
        size_t length); 

    /// Construct the FCI Mapping
    Spinmap build_mapping(
        const std::vector<uint64_t>& strings, 
        int nele, 
        const std::unordered_map<uint64_t, size_t>& index);

    std::vector<std::vector<std::vector<int>>> map_to_deexc(
        const Spinmap& mappings, 
        int states, 
        int norbs,
        int nele);

    std::vector<uint64_t> get_lex_bitstrings(int nele, int norb);

    uint64_t build_string_address(
        int nele, 
        int norb, 
        uint64_t occ,
        const std::vector<std::vector<uint64_t>>& zmat); 

    std::vector<std::vector<uint64_t>> get_z_matrix(int norb, int nele);

    /// ==> Utility Functions for Bit Math (may need to move) <== ///

    /// Combinutorics helper funciton for binomial coefficients
    int binom(int n, int m) {
        if (m < 0 || m > n)
            return 0;

        std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1, 0));
        for (int i = 0; i <= n; ++i)
            dp[i][0] = 1;

        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= std::min(i, m); ++j) {
                dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
            }
        }

        return dp[n][m];
    }

    std::vector<int> unroll_from_3d(
        const std::vector<std::vector<std::vector<int>>>& input) 
    {
        std::vector<int> output;
        
        for (const auto& vec2D : input) {
            for (const auto& vec1D : vec2D) {
                output.insert(output.end(), vec1D.begin(), vec1D.end());
            }
        }
        
        return output;
    }
    
    bool get_bit(uint64_t string, size_t pos) { return string & maskbit(pos); }

    constexpr uint64_t maskbit(size_t pos) { return static_cast<uint64_t>(1) << pos; }

    int count_bits_between(uint64_t string, int pos1, int pos2) {

        uint64_t mask = (((1 << pos1) - 1) ^ ((1 << (pos2 + 1)) - 1)) \
         & (((1 << pos2) - 1) ^ ((1 << (pos1 + 1)) - 1));

        uint64_t masked_string = string & mask;

        int count = 0;
        while (masked_string > 0) {
            count += masked_string & 1;
            masked_string >>= 1;
        }

        return count;
    }

    uint64_t set_bit(uint64_t string, int pos) {
        return string | (1ULL << pos);
    }

    uint64_t unset_bit(uint64_t string, int pos) {
        return string & ~(1ULL << pos);
    }

    /// ==> Setters and Getters <== /// 

    /// return the number of alfa/beta electrons
    size_t get_nalfa() const { return nalfa_; }
    size_t get_nbeta() const { return nbeta_; }


    /// return the alfa/beta bitstrings
    std::vector<uint64_t> get_astr() const { return astr_;  }
    std::vector<uint64_t> get_bstr() const { return bstr_;  }

    std::unordered_map<uint64_t, size_t> get_aind() const { return aind_; }
    std::unordered_map<uint64_t, size_t> get_bind() const { return bind_; }

    Spinmap get_alfa_map() const { return alfa_map_; }
    Spinmap get_beta_map() const { return beta_map_; }

    int get_ndexca() const { return alfa_map_.size(); }
    int get_ndexcb() const { return beta_map_.size(); }

    std::vector<std::vector<std::vector<int>>> get_dexca() const { return dexca_; }
    std::vector<std::vector<std::vector<int>>> get_dexcb() const { return dexcb_; }

    const std::vector<int>& read_dexca_vec() const { return dexca_vec_; }
    const std::vector<int>& read_dexcb_vec() const { return dexcb_vec_; }

private:
    int nalfa_;
    int nbeta_;
    int norb_;
    int lena_;
    int lenb_;

    std::vector<uint64_t> astr_;
    std::vector<uint64_t> bstr_;

    std::unordered_map<uint64_t, size_t> aind_;
    std::unordered_map<uint64_t, size_t> bind_;

    Spinmap alfa_map_;
    Spinmap beta_map_;

    std::vector<std::vector<std::vector<int>>> dexca_;
    std::vector<std::vector<std::vector<int>>> dexcb_;

    std::vector<int> dexca_vec_;
    std::vector<int> dexcb_vec_;
    
};

#endif
