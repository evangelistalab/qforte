#ifndef _sparse_tensor_h_
#define _sparse_tensor_h_

#include <complex>
#include <string>
#include <vector>
#include <map>

class SparseVector {
    /* A SparseVector is a custom class based on the standard map for sparse
     * vector storage and manipulation.
     * */
  public:
    /// default constructor: creates an empty sparse vector
    SparseVector() {}

    /// gets an element of the sparse vector. Returns the value for the inxex
    /// key I if key is contained in map, returns 0.0 otherwise
    std::complex<double> get_element(size_t I) const;

    /// returns the pointer (iterator) to the memory location of value corresponding
    /// to key I
    std::map<size_t, std::complex<double>>::const_iterator get_element_it(size_t I) const;

    /// returns the pointer (iterator) to the memory location of the last
    /// value in the map values_
    std::map<size_t, std::complex<double>>::const_iterator get_end() const;

    /// sets an element V_I and by appending the map values_
    void set_element(size_t I, std::complex<double> val);

    /// returns the SparseVector as a map
    std::map<size_t, std::complex<double>> to_map() const;

    /// clears key value pair corrsponding to key I
    void erase(size_t I);

  private:
    std::map<size_t, std::complex<double>> values_;
};

class SparseMatrix {
    /* A SparseMatrix is a custom class based on the standard map for sparse
     * matrix storage and manipulation.
     * */
  public:
    /// default constructor: creates an empty sparse matrix
    SparseMatrix() {}

    /// gets an element of the SparseMatrix, note that this function
    /// will return the value 0.0 if the I J pair is not a key for the
    /// 2d map.
    std::complex<double> get_element(size_t I, size_t J) const;

    /// sets an element M_IJ and by appending the 2d map
    void set_element(size_t I, size_t J, std::complex<double> val);

    /// returns a map with value type SparseVector
    std::map<size_t, SparseVector> to_vec_map() const;

    /// return the SparseSmatrix as a 2d map with keys as indices I and J, and
    /// a values pertianint to non-zero matrix elements M_IJ
    std::map<size_t, std::map<size_t, std::complex<double>>> to_map() const;

    /// get a list of unique J indicies for basis states
    std::vector<size_t> get_unique_js() const;

    /// multiply this matrix by Lmat such that this_new = Lmat x this_old
    void left_multiply(const SparseMatrix& Lmat);

    /// add a sparse matrix to this apply_matrix
    void add(const SparseMatrix& Mat, const std::complex<double> factor);

    /// makes this SparseMatrix into the identity.
    void make_identity(const size_t nbasis);

  private:
    std::map<size_t, SparseVector> values_;
};

#endif // _sparse_tensor_h_
