#ifndef _tensor_operator_h_
#define _tensor_operator_h_

#include <complex>
#include <string>
#include <vector>
#include <numeric>
#include <map>

// TODO(Tyler) temporary for debugging, remove when not needed
#include <iostream>

class SQOperator;
class Tensor;

class TensorOperator {
    /* A TensorOperator is a tensor represntation of various n-body operators
     * All orbital indices start at zero.
     */
  public:
    /// default constructor: creates an empty tensor operator
    TensorOperator(
      size_t max_nbody, 
      size_t dim, 
      bool is_spatial = false, 
      bool is_restricted = false
      );

    /// TODO(Nick): Implement all of these
    // /// adds a tensor operator to the second quantized operator
    // void add_top(const TensorOperator& to);

    /// adds a second quantized operator to the current tensor operator
    void add_sqop_of_rank(const SQOperator& sqo, const int);

    void add_sqop_term(const std::tuple< std::complex<double>, std::vector<size_t>, std::vector<size_t>>& sqo_term);

    // /// sets the tensor elements from another tensor operator
    // void set_from_tensor_op(const TensorOperator& to);

    // /// sets the tensor elements from a vector of tensors 
    // void set_from_tensor_list(const std::vector<Tensor>& ts);

    // /// sets elemets of a tensor with sepcified rank
    // void set_elements_of_rank(const Tensor& t, const size_t& rank );

    // /// multiplies all the tensor coefficients by multiplier
    // void scal(const std::complex<double>& multiplier);

    /// return a vector strings including 
    std::string str(
        bool print_data = true,
        bool print_complex = false,
        int maxcols = 5,
        const std::string& data_format = "%12.7f",
        const std::string& header_format = "%12zu"
        ) const;

    template<typename T>
    void print_vector(const std::string& str, const std::vector<T>& vec) {
        std::cout << str << std::endl;
        std::cout << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            std::cout << vec[i];
            if (i != vec.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
        // std::cout << std::endl << std::endl;
    }

  private:
    /// The linear combination of second quantized operators. Stored as a tuple of
    std::vector<Tensor> tensors_;

    /// The highest order n-body excitation 
    size_t max_nbody_; 

    /// The dimesion of the tensors, for example 
    /// if dim is specified as 4 then the follwoing shapes would be returned:
    /// tensors_[0].shape => (0,) [just a scalar always]
    /// tensors_[1].shape => (4, 4) [just a scalar always]
    /// tensors_[2].shape => (4, 4, 4, 4) [just a scalar always]
    /// Also equal to the number of spin orbs if is_spatial_ == false,
    /// or the number of spatial orbs if is_spatial_ == false,
    size_t dim_;

    /// Whether the tensors correspond to spatial orbitals
    /// if false it is assumed that all indices correspond to qubits/spin-orbitals
    bool is_spatial_;  

    /// Whether the elements (integrals) come from a restricted calculation
    /// Should obly be true if is_spatial_ is also true 
    bool is_restricted_; 
};

#endif // _sq_operator_h_
