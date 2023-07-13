#ifndef _tensor_h_
#define _tensor_h_

#include <memory>
#include <cstddef>
#include <string>
#include <vector>

#include "qforte-def.h"

// namespace lightspeed { 

/**
 * Class Tensor contains a simple tensor class with core-only capabilities.
 * Tensor objects are rectilinear double precision tensors of arbitrary number
 * of dimensions (including scalars). The data in a Tensor object is always
 * stored in row-major order (C-order), with the last dimension stored in
 * contiguous order. E.g., for a matrix, the columns are stored contiguously,
 * the rows are strided across the column dimension.
 **/ 
// class Tensor final {
class Tensor {

public:

// => Constructors <= //

/**
 * Constructor: Builds and initializes the Tensor to all zeros.
 *
 * @param shape the shape of the tensor
 * @param name name of the tensor (for printing/filename use)
 **/
Tensor(
    const std::vector<size_t>& shape,
    const std::string& name = "T"
    );

~Tensor();

// => Topology Accessors <= //

/// The name of this Tensor
std::string name() const { return name_; }

/// The number of dimensions of this Tensor, inferred from shape
size_t ndim() const { return shape_.size(); }

/// The total number of elements of this Tensor (the product of shape)
size_t size() const { return size_; }

/// The size in each dimension of this Tensor
const std::vector<size_t>& shape() const { return shape_; }

/// The offset between consecutive indices within each dimension
const std::vector<size_t>& strides() const { return strides_; }

// => Data Accessors <= //

/**
 * The data of this Tensor, using C-style compound indexing. Modifying the
 * elements of this vector will modify the data of this Tensor
 *
 * @return a reference to the vector data of this tensor
 **/
std::vector<std::complex<double>>& data() { return data_; }

/**
 * The data of this Tensor, using C-style compound indexing.
 *
 * @return a reference to the vector data of this tensor, can't be modified, only read
 **/
const std::vector<std::complex<double>>& read_data() const { return data_; }

// => Setters <= //

/// Set this Tensor's name to @param name
void set_name(const std::string& name) { name_ = name; } 

// => Clone Actions <= //

/// Create a new copy of this Tensor (same size and data)
std::shared_ptr<Tensor> clone();

/// Set a particular element of tis Tensor, specified by idxs
void set(const std::vector<size_t>& idxs,
         const std::complex<double> val
         );

/// Get a particular element of tis Tensor, specified by idxs
std::complex<double> get(const std::vector<size_t>& idxs) const;

/// Get the vector index for this tensor based on the tensor index
size_t tidx_to_vidx(const std::vector<size_t>& tidx) const;

/// Get the vector index for this tensor based on the tensor index, and axes
size_t tidx_to_trans_vidx(const std::vector<size_t>& tidx, const std::vector<size_t>& axes) const;

/// Get the tensor index for this tensor based on the vector index
std::vector<size_t> vidx_to_tidx(size_t vidx) const;

// => Simple Core Actions <= //

/**
 * Set all elements of this Tensor to zero
 **/
void zero();

/**
 * Set this 2D square Tensor to the identity matrix
 * Throw if not 2D square
 **/
void identity();

/**
 * Set this 2D square Tensor T to 0.5 * (T + T')
 * Throw if not 2D square
 **/
void symmetrize();

/**
 * Set this 2D square Tensor T to 0.5 * (T - T')
 * Throw if not 2D square
 **/
void antisymmetrize();

/**
 * Scale this Tensor by param a
 * @param a the scalar multiplier
 **/
void scale(std::complex<double> a);

/**
 * Copy the data of Tensor other to this Tensor
 * @param other Tensor to copy data from
 * Throw if other is not same shape 
 * TODO: This is covered by a static Python method, deprecate and remove this function.
 **/
void copy(const std::shared_ptr<Tensor>& other); 

/**
 * Update this Tensor (y) to be y = a * x + b * y
 * Throw if x is not same shape 
 **/
void axpby(const std::shared_ptr<Tensor>& x, double a, double b);

/**
 * Update this Tensor (y) to be y = x + y
 * Throw if x is not same shape 
 **/
void add(const Tensor& x);

/**
 * Compute the dot product between this and other Tensors,
 * by unrolling this and other Tensor and adding sum of products of
 * elements
 *
 * @param other Tensor to take dot product with
 * @return the dot product
 * Throw if other is not same shape 
 **/
double vector_dot(const std::shared_ptr<Tensor>& other) const;

/**
 * Compute a new copy of this Tensor which is a transpose of this. Works only
 * for matrices. 
 *
 * @return a transposed copy of this
 * Throw if not 2 ndim
 **/
Tensor transpose() const;

/**
 * Compute a new copy of this Tensor which is a transpose of this.
 *
 * @return a transposed copy of this acording to axes
 **/
Tensor general_transpose(const std::vector<size_t>& axes) const;

// => Printing <= //

/**
 * Return a string representation of this Tensor
 * @param print_data print the data (true) or size info only (false)? 
 * @param maxcols the maximum number of columns to print before
 *  going to a new block
 * @param data_format the format for data printing
 * @param header_format the format of the column index
 * @return the string form of the tensor 
 **/
std::string str(
    bool print_data = true,
    bool print_complex = false,
    int maxcols = 5,
    const std::string& data_format = "%12.7f",
    const std::string& header_format = "%12zu"
    ) const; 

void fill_from_nparray(std::vector<std::complex<double>>, std::vector<size_t>);

/**
 * Print string representation of this Tensor
 **/
// void print() const;

/**
 * Print string representation of this Tensor with name
 **/
// void print(const std::string& name);

// => Error Throwers <= //

/**
 * Throw std::runtime error if ndim() != ndim
 **/
void ndim_error(size_t ndim) const;

/**
 * Throw std::runtime_error if shape != shape()
 * First calls ndim_error(shape_.size())
 **/
void shape_error(const std::vector<size_t>& shape) const;

/**
 * Throw std::runtime_error if not square matrix
 * First calls ndim_error(2)
 **/
void square_error() const;

/// NICK: Comment out the functions below for now, will need external lib
// => Tensor Multiplication/Permutation <= //

// /**
//  * Performed the chained matrix multiplication:
//  *      
//  *  C = alpha * As[0]^trans[0] * As[1]^trans[1] * ... + beta * C
//  *      
//  *  @param As the list of A core Tensors
//  *  @param trans the list of transpose arguments
//  *  @param C the resultant matrix - if this argument is not provided, C is
//  *      allocated and set to zero in the routine
//  *  @param alpha the prefactor of the chained multiply
//  *  @param beta the prefactor of the register tensor C
//  *  @return C - the resultant tensor (for chaining and new allocation)
//  **/
// static std::shared_ptr<Tensor> chain(
//     const std::vector<std::shared_ptr<Tensor> >& As,
//     const std::vector<bool>& trans,
//     const std::shared_ptr<Tensor>& C = std::shared_ptr<Tensor>(),
//     double alpha = 1.0,
//     double beta = 0.0);

// static std::shared_ptr<Tensor> permute(
//     const std::vector<std::string>& Ainds,
//     const std::vector<std::string>& Cinds,
//     const std::shared_ptr<Tensor>& A,
//     const std::shared_ptr<Tensor>& C = std::shared_ptr<Tensor>(),
//     double alpha = 1.0,
//     double beta = 0.0);

// static std::shared_ptr<Tensor> einsum(
//     const std::vector<std::string>& Ainds,
//     const std::vector<std::string>& Binds,
//     const std::vector<std::string>& Cinds,
//     const std::shared_ptr<Tensor>& A,
//     const std::shared_ptr<Tensor>& B,
//     const std::shared_ptr<Tensor>& C = std::shared_ptr<Tensor>(),
//     double alpha = 1.0,
//     double beta = 0.0);

// // => Linear Algebra <= //

// // Returns in C-order
// static std::shared_ptr<Tensor> potrf(
//     const std::shared_ptr<Tensor>& S,
//     bool lower = true);

// // Returns in C-order
// static std::shared_ptr<Tensor> trtri(
//     const std::shared_ptr<Tensor>& L,
//     bool lower = true);

// // Works in F-order
// static std::shared_ptr<Tensor> gesv(
//     const std::shared_ptr<Tensor>& A,
//     std::shared_ptr<Tensor>& f);

// // Returns in C-order (A = U a U')
// static void syev(
//     const std::shared_ptr<Tensor>& A,
//     std::shared_ptr<Tensor>& U,
//     std::shared_ptr<Tensor>& a,
//     bool ascending = true,
//     bool syevd = true);

// // Returns in C-order (A = U a U') using Cholesky factorization
// static void generalized_syev(
//     const std::shared_ptr<Tensor>& A,
//     const std::shared_ptr<Tensor>& S,
//     std::shared_ptr<Tensor>& U,
//     std::shared_ptr<Tensor>& a,
//     bool ascending = true,
//     bool syevd = true);

// // Returns in C-order (A = U a U') using Canonical orthogonalization
// static void generalized_syev2(
//     const std::shared_ptr<Tensor>& A,
//     const std::shared_ptr<Tensor>& S,
//     std::shared_ptr<Tensor>& U,
//     std::shared_ptr<Tensor>& a,
//     bool ascending = true,
//     bool syevd = true,
//     double tolerance=1.0E-9);

// // Returns in C-order (A = U s V)
// static void gesvd(
//     const std::shared_ptr<Tensor>& A,
//     std::shared_ptr<Tensor>& U,
//     std::shared_ptr<Tensor>& s,
//     std::shared_ptr<Tensor>& V,
//     bool full_matrices = false,
//     bool gesdd = true);

// static std::shared_ptr<Tensor> power(
//     const std::shared_ptr<Tensor>& S,
//     double power = -1.0,
//     double condition = 1.0E-10,
//     bool throwNaN = false);

// // Returns in C-order (1 = X' S X)
// static std::shared_ptr<Tensor> lowdin_orthogonalize(
//     const std::shared_ptr<Tensor>& S);

// // Returns in C-order (1 = X' S X)
// static std::shared_ptr<Tensor> cholesky_orthogonalize(
//     const std::shared_ptr<Tensor>& S);

// // Returns in C-order (1 = X' S X)
// static std::shared_ptr<Tensor> canonical_orthogonalize(
//     const std::shared_ptr<Tensor>& S,
//     double condition = 1.0E-10);

// /**
//  * Invert the Tensor in place via LU decomposition. Returns the determinant of
//  * the original matrix. If a zero pivot is indicated by DGETRF (indicating zero
//  * determinant), the code returns immediately without calling DGETRI, and the
//  * contents of the matrix are the result of the call to DGETRF.
//  *
//  * @return D the determinant of the original matrix
//  * @result the inverse of the matrix is formed in place, unless a zero
//  *   determinant is detected in which case the result of DGETRF is formed in
//  *   place.
//  **/
// double invert_lu();

private:

std::string name_;

std::vector<size_t> shape_;

std::vector<size_t> strides_;

size_t size_;

/// TODO(Nick): I am sure this will cause problems...
std::vector<std::complex<double>> data_;

// => Ed's special total memory thing <= //

private: 

static size_t total_memory__;

public:

/**
 * Current total global memory usage of Tensor in bytes. 
 * Computes as t.size() * sizeof(double) for all tensors t that are currently
 * in scope.
 **/
static size_t total_memory() { return total_memory__; }

};

// } // namespace lightspeed

#endif // _tensor_h_