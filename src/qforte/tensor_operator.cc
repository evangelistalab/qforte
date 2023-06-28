#include <algorithm>
#include <stdexcept>
#include <tuple>

#include "helpers.h"
// #include "gate.h"
// #include "circuit.h"
// #include "qubit_operator.h"
#include "sq_operator.h"
#include "tensor_operator.h"
#include "tensor.h"

TensorOperator::TensorOperator(
    size_t max_nbody, 
    size_t dim, 
    bool is_spatial, 
    bool is_restricted
        ) : 
    max_nbody_(max_nbody), 
    dim_(dim),
    is_spatial_(is_spatial),
    is_restricted_(is_restricted) 
{   
    if(max_nbody_ < 0){throw std::invalid_argument( "max_nbody must be > 0" );}
    if(max_nbody_ > 4){throw std::invalid_argument( "max_nbody must be <= 4" );}
    if(is_restricted_ and not is_spatial_){
        throw std::invalid_argument( "if using restricted calculation, must also use spatial orbital indexing" );
    }

    // Initialize the tensors 

    std::vector<size_t> shape0 = {1, 1};
    std::string name0 = "zero-body";
    Tensor zero_body(shape0, name0);

    // for (size_t i = 0; i < max_nbody_; i++) {
    //     basis_[i] = QubitBasis(i);
    // }
    tensors_.push_back(zero_body);
}

// void TensorOperator::add_top(const TensorOperator& qo) {
//     terms_.push_back(std::make_tuple(circ_coeff, cre_ops, ann_ops));
// }

// void TensorOperator::add_sqop(const SQOperator& qo) {
//     terms_.insert(terms_.end(), qo.terms().begin(), qo.terms().end());
// }

// void TensorOperator::set_from_tensor_op(const TensorOperator& to) {
//     if(new_coeffs.size() != terms_.size()){
//         throw std::invalid_argument( "number of new coefficients for quantum operator must equal " );
//     }
//     for (auto l = 0; l < new_coeffs.size(); l++) {
//         std::get<0>(terms_[l]) = new_coeffs[l];
//     }
// }

// void TensorOperator::set_from_tensor_list(const std::vector<Tensor>& ts) {
//     if(new_coeffs.size() != terms_.size()){
//         throw std::invalid_argument( "number of new coefficients for quantum operator must equal " );
//     }
//     for (auto l = 0; l < new_coeffs.size(); l++) {
//         std::get<0>(terms_[l]) = new_coeffs[l];
//     }
// }

// void TensorOperator::set_elements_of_rank(const Tensor& t, const size_t& rank ) {
//     if(new_coeffs.size() != terms_.size()){
//         throw std::invalid_argument( "number of new coefficients for quantum operator must equal " );
//     }
//     for (auto l = 0; l < new_coeffs.size(); l++) {
//         std::get<0>(terms_[l]) = new_coeffs[l];
//     }
// }

// void TensorOperator::scal(const std::complex<double>& multiplier) {
//     for (auto& term : terms_){
//         std::get<0>(term) *= multiplier;
//     }
// }

std::string TensorOperator::str(
    bool print_data, 
    bool print_complex, 
    int maxcols,
    const std::string& data_format,
    const std::string& header_format
    ) const 
{
    std::vector<std::string> s;
    s.push_back("");
    for (const auto& tensor : tensors_) {
        s.push_back(
            tensor.str(
                print_data, 
                print_complex, 
                maxcols,
                data_format,
                header_format
            )
        );
        s.push_back(")\n\n");
    }
    return join(s, " ");
}
