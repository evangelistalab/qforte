#include <algorithm>
#include <stdexcept>
#include <tuple>

#include "helpers.h"
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

    // Initialize the scaler (zero-body) "Tensor"
    std::vector<size_t> shape0 = {1};
    std::string name0 = "0-body";
    Tensor zero_body(shape0, name0);
    tensors_.push_back(zero_body);

    // Initialize the (n-body) Tensors
    for (size_t i = 1; i < max_nbody_ + 1; i++) {
        std::vector<size_t> shape_n;
        for (size_t j = 0; j < 2*i; j++){shape_n.push_back(dim_);}
        std::string name_n = std::to_string(i) + "-body";
        Tensor n_body(shape_n, name_n);
        tensors_.push_back(n_body);
    }
    
}

// void TensorOperator::add_top(const TensorOperator& qo) {
//     terms_.push_back(std::make_tuple(circ_coeff, cre_ops, ann_ops));
// }

void TensorOperator::add_sqop_of_rank(const SQOperator& sqo, const int rank) {
    if (is_spatial_ or is_restricted_) {
        throw std::invalid_argument("Can only add SQOperator if TensorOperator is not spatial");
    }

    // vector of ranks present
    std::vector<int> ranks_present = sqo.ranks_present();
    if (ranks_present.size() != 1) {
        throw std::invalid_argument("More than one rank dectected in SQOperter");
    }

    if (ranks_present[0] != rank) {
        throw std::invalid_argument("Rank of SQOperter does not match rank requested to add");
    }

    // largest alfa and beta orbital indices
    std::pair<int, int> abmax_idxs = sqo.get_largest_alfa_beta_indices(); 

    if (static_cast<int>(dim_) <= abmax_idxs.first) {
        throw std::invalid_argument("Highest alpha index exceeds the norb of orbitals");
    }
    if (static_cast<int>(dim_) <= abmax_idxs.second) {
        throw std::invalid_argument("Highest beta index exceeds the norb of orbitals");
    }

    int rank_index = rank / 2;

    if (rank % 2) {
        throw std::invalid_argument("Odd rank operator not supported");
    }

    if (rank < 0) {
        throw std::invalid_argument("Can't have a negative rank");
    }

    if (rank > 2 * max_nbody_) {
        throw std::invalid_argument("Trying to add SQOperator of higher rank than permitted by max nbody");
    }

    if (rank == 0) {
        std::complex<double> h0e = 0.0;
        for (const auto& term : sqo.terms()) {
            h0e += std::get<0>(term);
        }
        tensors_[0].set({0}, h0e);

    } else {
        // dimensions of tensor, ex [dim, dim, dim, dim] for a 2-body operator
        std::vector<size_t> tensor_dim(rank, dim_);
        std::vector<size_t> index_mask(rank, 0);

        std::vector<std::vector<int>> index_dict_dagger(std::floor(rank / 2), std::vector<int>(2, 0));
        std::vector<std::vector<int>> index_dict_nondagger(std::floor(rank / 2), std::vector<int>(2, 0));

        // each n-body operator makes a contribution
        for (const auto& term : sqo.terms()) {

            if (std::get<1>(term).size() != std::get<2>(term).size()) {
                throw std::invalid_argument("Each term must have same number of anihilators and creators");
            }   

            std::vector<size_t> ops1(std::get<1>(term));
            std::vector<size_t> ops2(std::get<2>(term));
            ops1.insert(ops1.end(), ops2.begin(), ops2.end());

            for (int i = 0; i < rank; ++i) {
                // index of anihilator or creator
                int index = ops1[i];

                int spin = index % 2;
                int ind;

                if (spin == 1) {
                    ind = (index - 1) / 2 + (dim_ / 2);
                }
                else {
                    ind = index / 2;
                }

                if (i < rank / 2) {
                    index_dict_dagger[i][0] = spin;
                    index_dict_dagger[i][1] = ind;
                }
                else {
                    index_dict_nondagger[i - rank / 2][0] = spin;
                    index_dict_nondagger[i - rank / 2][1] = ind;
                }
            }

            // May or may not need is if operators are already canonicalized
            int parity = reverse_bubble_list(index_dict_dagger);
            parity += reverse_bubble_list(index_dict_nondagger);

            for (int i = 0; i < rank; ++i) {
                if (i < rank / 2) {
                    index_mask[i] = index_dict_dagger[i][1];
                }
                else {
                    index_mask[i] = index_dict_nondagger[i - rank / 2][1];
                }
            }

            std::complex<double> val = tensors_[rank_index].get(index_mask); 
            val += pow(-1, parity) * std::get<0>(term);
            tensors_[rank_index].set(index_mask, val); 
        }

        Tensor tensor2(tensors_[rank_index].shape(), "T2");
        double length = 0.0;
        std::vector<size_t> seed(rank / 2);
        for (size_t i = 0; i < rank / 2; ++i) {
            seed[i] = i;
        }

        do {
            std::vector<size_t> iperm(seed);
            std::vector<size_t> jperm(seed);

            for (size_t j = 0; j < rank / 2; ++j) {
                jperm[j] += rank / 2;
            }

            iperm.insert(iperm.end(), jperm.begin(), jperm.end());

            Tensor transposed_tensor = tensors_[rank_index].general_transpose(iperm);
            tensor2.add(transposed_tensor);
            length += 1.0;
        } while (next_permutation(seed.begin(), seed.end()));

        tensor2.scale(1.0/length);

        tensors_[rank_index] = tensor2;
    }
}

void TensorOperator::fill_tensor_from_np_by_rank(int idx, std::vector<std::complex<double>> arr, std::vector<size_t> shape){

    if (idx % 2 != 0){
        throw std::runtime_error("Invalid Rank.");
    }

    if (arr.size() != tensors_[idx/2].size()){
        throw std::runtime_error("Sizes of arrays are not the same.");
    }

    tensors_[idx/2].fill_from_nparray(arr, shape);

}


// void TensorOperator::add_sqop_term(const std::tuple< std::complex<double>, std::vector<size_t>, std::vector<size_t>>& sqo_term )
// {

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

// TODO(Nick/Tyler) This function prints a bunch of extr lines in 
// Python, not sure why, would like to fix
std::string TensorOperator::str(
    bool print_data, 
    bool print_complex, 
    int maxcols,
    const std::string& data_format,
    const std::string& header_format
    ) const 
{
    std::string s;
    for (const auto& tensor : tensors_) {
        s += tensor.str(
                print_data, 
                print_complex, 
                maxcols,
                data_format,
                header_format
            );
        s += std::printf("\n");
    }
    return s;
}
