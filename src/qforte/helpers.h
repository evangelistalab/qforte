#include <complex>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

std::string join(const std::vector<std::string>& vec_str, const std::string& sep = ",");

std::string to_string(std::complex<double> value);

/// returns the parity given a list of creators and anihliators
int reverse_bubble_list(std::vector<std::vector<int>>& arr);

// class SparseVector {
//     /* A SparseVector is a custom class based on the standard map for sparse
//      * vector storage and manipulation.
//      * */
//   public:
//     /// default constructor: creates an empty sparse vector
//     SparseVector() {}
//
//     /// gets an element of the sparse vector. Returns the value for the inxex
//     /// key if key is contained in map, returns zero otherwise.
//     std::complex<double> get_element(size_t I) const {
//         std::map<size_t, std::complex<double>>::const_iterator i = values_.find(I);
//         if(i==values_.end()) {
//             return 0.0;
//         }
//         return i->second;
//     }
//
//     std::map<size_t, std::complex<double>>::const_iterator get_element_it(size_t I) const {
//         std::map<size_t, std::complex<double>>::const_iterator i = values_.find(I);
//         return i;
//     }
//
//     std::map<size_t, std::complex<double>>::const_iterator get_end() const {
//         return values_.end();
//     }
//
//     void set_element(size_t I, std::complex<double> val) {
//         values_[I] = val;
//     }
//
//     std::map<size_t, std::complex<double>> to_map() const {
//         return values_;
//     }
//
//   private:
//     std::map<size_t, std::complex<double>> values_;
//
// };
//
// class SparseMatrix {
//     /* A SparseMatrix is a custom class based on the standard map for sparse
//      * matrix storage and manipulation.
//      * */
//   public:
//     /// default constructor: creates an empty sparse matrix
//     SparseMatrix() {}
//
//     std::complex<double> get_element(size_t I, size_t J) const {
//         std::map<size_t, SparseVector>::const_iterator i = values_.find(I);
//         if(i==values_.end()) {
//             return 0.0;
//         }
//         return i->second.get_element(J);
//     }
//
//     void set_element(size_t I, size_t J, std::complex<double> val) {
//         values_[I].set_element(J, val);
//     }
//
//     std::map<size_t, std::map<size_t, std::complex<double>>> to_map() const {
//         std::map<size_t, std::map<size_t, std::complex<double>>> sp_mat_map;
//
//         for (auto const& x : values_){
//             sp_mat_map.insert(std::make_pair(x.first, x.second.to_map()));
//         }
//         return sp_mat_map;
//     }
//
//   private:
//     std::map<size_t, SparseVector> values_;
// };
