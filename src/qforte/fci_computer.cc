#include <map>
#include <random>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <cmath>

// #include "fmt/format.h"

#include "qubit_basis.h"
#include "circuit.h"
#include "gate.h"
#include "helpers.h"
#include "qubit_operator.h"
#include "tensor.h"
#include "tensor_operator.h"
#include "qubit_op_pool.h"
#include "timer.h"
#include "sq_operator.h"
#include "blas_math.h"

#include "fci_computer.h"
#include "fci_graph.h"


FCIComputer::FCIComputer(int nel, int sz, int norb) : 
    nel_(nel), 
    sz_(sz),
    norb_(norb) {

    if (nel_ < 0) {
        throw std::invalid_argument("Cannot have negative electrons");
    }
    if (nel_ < std::abs(static_cast<double>(sz_))) {
        throw std::invalid_argument("Spin quantum number exceeds physical limits");
    }
    if ((nel_ + sz_) % 2 != 0) {
        throw std::invalid_argument("Parity of spin quantum number and number of electrons is incompatible");
    }

    nalfa_el_ = (nel_ + sz_) / 2;
    nbeta_el_ = nel_ - nalfa_el_;

    nalfa_strs_ = 1;
    for (int i = 1; i <= nalfa_el_; ++i) {
        nalfa_strs_ *= norb_ - i + 1;
        nalfa_strs_ /= i;
    }

    if (nalfa_el_ < 0 || nalfa_el_ > norb_) {
        nalfa_strs_ = 0;
    }

    nbeta_strs_ = 1;
    for (int i = 1; i <= nbeta_el_; ++i) {
        nbeta_strs_ *= norb_ - i + 1;
        nbeta_strs_ /= i;
    }

    if (nbeta_el_ < 0 || nbeta_el_ > norb_) {
        nbeta_strs_ = 0;
    }

    C_.zero_with_shape({nalfa_strs_, nbeta_strs_});
    C_.set_name("FCI Computer");

    graph_ = FCIGraph(nalfa_el_, nbeta_el_, norb_);
}

/// apply a TensorOperator to the current state 
void apply_tensor_operator(const TensorOperator& top);

/// apply a Tensor represending a 1-body spin-orbital indexed operator to the current state 
void FCIComputer::apply_tensor_spin_1bdy(const Tensor& h1e, size_t norb) {

    if(h1e.size() != (norb * 2) * (norb * 2)){
        throw std::invalid_argument("Expecting h1e to be nso x nso for apply_tensor_spin_1bdy");
    }

    Tensor Cnew({nalfa_strs_, nbeta_strs_}, "Cnew");

    Tensor h1e_blk1 = h1e.slice(
        {
            std::make_pair(0, norb_), 
            std::make_pair(0, norb_)
            }
        );

    Tensor h1e_blk2 = h1e.slice(
        {
            std::make_pair(norb_, 2*norb_), 
            std::make_pair(norb_, 2*norb_)
            }
        );

    apply_array_1bdy(
        Cnew,
        graph_.read_dexca_vec(),
        nalfa_strs_,
        nbeta_strs_,
        graph_.get_ndexca(),
        h1e_blk1,
        norb_,
        true);

    apply_array_1bdy(
        Cnew,
        graph_.read_dexcb_vec(),
        nalfa_strs_,
        nbeta_strs_,
        graph_.get_ndexcb(),
        h1e_blk2,
        norb_,
        false);

    C_ = Cnew;
}

/// apply Tensors represending 1-body and 2-body spin-orbital indexed operator to the current state 
/// A LOT of wasted memory here, will want to improve...
void FCIComputer::apply_tensor_spin_12bdy(
    const Tensor& h1e, 
    const Tensor& h2e, 
    size_t norb) 
{
    if(norb > 10 or norb_ > 10){
        throw std::invalid_argument("Don't use this function with more that 10 orbitals, too much memory");
    }

    if(h1e.size() != (norb * 2) * (norb * 2)){
        throw std::invalid_argument("Expecting h1e to be nso x nso for apply_tensor_spin_12bdy");
    }

    if(h2e.size() != (norb * 2) * (norb * 2) * (norb * 2) * (norb * 2) ){
        throw std::invalid_argument("Expecting h2e to be nso x nso x nso nso for apply_tensor_spin_12bdy");
    }

    Tensor Cnew({nalfa_strs_, nbeta_strs_}, "Cnew");
    Tensor h1e_new = h1e;

    Tensor h2e_new(h2e.shape(), "A2");
    Tensor::permute(
        {"i", "j", "k", "l"}, 
        {"i", "k", "j", "l"}, 
        h2e, 
        h2e_new); 

    h2e_new.scale(-1.0);

    for(int k = 0; k < 2 * norb_; k++){
        Tensor h2e_k = h2e.slice(
        {
            std::make_pair(0, 2 * norb_), 
            std::make_pair(k, k+1),
            std::make_pair(k, k+1), 
            std::make_pair(0, 2 * norb_)
            }
        );

        h2e_k.scale(-1.0);

        h1e_new.zaxpy(
            h2e_k,
            1.0,
            1,
            1);
    }

    /// NICK: Keeping this here in case future debuggin is needed
    // 1 std::cout << "\n\n  ====> h1e <====" << h1e_new.print_nonzero() << std::endl;
    // 1 std::cout << "\n\n  ====> h2e <====" << h2e_new.print_nonzero() << std::endl;

    Tensor h1e_blk1 = h1e_new.slice(
        {
            std::make_pair(0, norb_), 
            std::make_pair(0, norb_)
            }
        );

    Tensor h1e_blk2 = h1e_new.slice(
        {
            std::make_pair(norb_, 2*norb_), 
            std::make_pair(norb_, 2*norb_)
            }
        );

    Tensor h2e_blk11 = h2e_new.slice(
        {
            std::make_pair(0, norb_), 
            std::make_pair(0, norb_),
            std::make_pair(0, norb_), 
            std::make_pair(0, norb_)
            }
        );

    Tensor h2e_blk12 = h2e_new.slice(
        {
            std::make_pair(0, norb_), 
            std::make_pair(0, norb_),
            std::make_pair(norb_, 2*norb_), 
            std::make_pair(norb_, 2*norb_)
            }
        );

    Tensor h2e_blk21 = h2e_new.slice(
        {
            std::make_pair(norb_, 2*norb_), 
            std::make_pair(norb_, 2*norb_),
            std::make_pair(0, norb_), 
            std::make_pair(0, norb_)
            }
        );

    Tensor h2e_blk22 = h2e_new.slice(
        {
            std::make_pair(norb_, 2*norb_), 
            std::make_pair(norb_, 2*norb_),
            std::make_pair(norb_, 2*norb_), 
            std::make_pair(norb_, 2*norb_)
            }
        );

    std::pair<Tensor, Tensor> dvec = calculate_dvec_spin_with_coeff();
    
    Tensor dveca_new(dvec.first.shape(),  "dveca_new");
    Tensor dvecb_new(dvec.second.shape(), "dvecb_new");

    Tensor::einsum(
        {"i", "j"},
        {"i", "j", "k", "l"},
        {"k", "l"},
        h1e_blk1,
        dvec.first,
        Cnew, 
        1.0,
        0.0
    );

    Tensor::einsum(
        {"i", "j"},
        {"i", "j", "k", "l"},
        {"k", "l"},
        h1e_blk2,
        dvec.second,
        Cnew, 
        1.0,
        0.0
    );

    Tensor::einsum(
        {"i", "j", "k", "l"},
        {"k", "l", "m", "n"},
        {"i", "j", "m", "n"},
        h2e_blk11,
        dvec.first,
        dveca_new, 
        1.0,
        0.0
    );

    Tensor::einsum(
        {"i", "j", "k", "l"},
        {"k", "l", "m", "n"},
        {"i", "j", "m", "n"},
        h2e_blk12,
        dvec.second,
        dveca_new, 
        1.0,
        0.0
    );

    Tensor::einsum(
        {"i", "j", "k", "l"},
        {"k", "l", "m", "n"},
        {"i", "j", "m", "n"},
        h2e_blk21,
        dvec.first,
        dvecb_new, 
        1.0,
        0.0
    );

    Tensor::einsum(
        {"i", "j", "k", "l"},
        {"k", "l", "m", "n"},
        {"i", "j", "m", "n"},
        h2e_blk22,
        dvec.second,
        dvecb_new, 
        1.0,
        0.0
    );

    std::pair<Tensor, Tensor> dvec_new = std::make_pair(dveca_new, dvecb_new);

    Cnew.zaxpy(
        calculate_coeff_spin_with_dvec(dvec_new),
        1.0,
        1,
        1    
    );

    C_ = Cnew;
}

/// apply Tensors represending 1-body and 2-body spin-orbital indexed operator to the current state 
void FCIComputer::apply_tensor_spin_012bdy(
    const Tensor& h0e, 
    const Tensor& h1e, 
    const Tensor& h2e, 
    size_t norb) 
{
    h0e.shape_error({1});
    Tensor Czero = C_;
    
    apply_tensor_spin_12bdy(
        h1e,
        h2e,
        norb);

    C_.zaxpy(
        Czero,
        h0e.get({0}),
        1,
        1    
    );
}

// NICK: VERY VERY Slow, will want even a better c++ implementation!
// Try with einsum once working or perhaps someting like the above...?
std::pair<Tensor, Tensor> FCIComputer::calculate_dvec_spin_with_coeff() {

    Tensor dveca({norb_, norb_, nalfa_strs_, nbeta_strs_}, "dveca");
    Tensor dvecb({norb_, norb_, nalfa_strs_, nbeta_strs_}, "dvecb");

    for (size_t i = 0; i < norb_; ++i) {
        for (size_t j = 0; j < norb_; ++j) {
            auto alfa_mappings = graph_.get_alfa_map()[std::make_pair(i,j)];
            auto beta_mappings = graph_.get_beta_map()[std::make_pair(i,j)];

            for (const auto& mapping : alfa_mappings) {
                size_t source = std::get<0>(mapping);
                size_t target = std::get<1>(mapping);
                std::complex<double> parity = static_cast<std::complex<double>>(std::get<2>(mapping));
                for (size_t k = 0; k < dveca.shape()[3]; ++k) {
                    size_t c_vidxa = k * C_.strides()[1] + source * C_.strides()[0];
                    size_t d_vidxa = k * dveca.strides()[3] + target * dveca.strides()[2] + j * dveca.strides()[1] + i * dveca.strides()[0];
                    dveca.data()[d_vidxa] += parity * C_.data()[c_vidxa];
                }
            }

            for (const auto& mapping : beta_mappings) {
                size_t source = std::get<0>(mapping);
                size_t target = std::get<1>(mapping);
                std::complex<double> parity = static_cast<std::complex<double>>(std::get<2>(mapping));
                for (size_t k = 0; k < dvecb.shape()[2]; ++k) {
                    size_t c_vidxb = source * C_.strides()[1] + k * C_.strides()[0];
                    size_t d_vidxb = target * dvecb.strides()[3] + k * dvecb.strides()[2] + j * dvecb.strides()[1] + i * dvecb.strides()[0];
                    dvecb.data()[d_vidxb] += parity * C_.data()[c_vidxb];
                }
            }
        }
    }
    return std::make_pair(dveca, dvecb);
}

// ALSO SLOW
Tensor FCIComputer::calculate_coeff_spin_with_dvec(std::pair<Tensor, Tensor>& dvec) {
    Tensor Cnew({nalfa_strs_, nbeta_strs_}, "Cnew");

    for (size_t i = 0; i < norb_; ++i) {
        for (size_t j = 0; j < norb_; ++j) {

            auto alfa_mappings = graph_.get_alfa_map()[std::make_pair(j,i)];
            auto beta_mappings = graph_.get_beta_map()[std::make_pair(j,i)];

            for (const auto& mapping : alfa_mappings) {
                size_t source = std::get<0>(mapping);
                size_t target = std::get<1>(mapping);
                std::complex<double> parity = static_cast<std::complex<double>>(std::get<2>(mapping));
                for (size_t k = 0; k < dvec.first.shape()[3]; ++k) {
                    size_t c_vidxa = k * Cnew.strides()[1] + source * Cnew.strides()[0];
                    size_t d_vidxa = k * dvec.first.strides()[3] + target * dvec.first.strides()[2] + j * dvec.first.strides()[1] + i * dvec.first.strides()[0];
                    Cnew.data()[c_vidxa] += parity * dvec.first.data()[d_vidxa];
                }
                
            }
            for (const auto& mapping : beta_mappings) {
                size_t source = std::get<0>(mapping);
                size_t target = std::get<1>(mapping);
                std::complex<double> parity = static_cast<std::complex<double>>(std::get<2>(mapping));
                for (size_t k = 0; k < dvec.second.shape()[2]; ++k) {
                    size_t c_vidxb = source * Cnew.strides()[1] + k * Cnew.strides()[0];
                    size_t d_vidxb = target * dvec.second.strides()[3] + k * dvec.second.strides()[2] + j * dvec.second.strides()[1] + i * dvec.second.strides()[0];
                    Cnew.data()[c_vidxb] += parity * dvec.second.data()[d_vidxb];
                }
            }
        }
    }

    return Cnew;
}

void FCIComputer::apply_array_1bdy(
    Tensor& out,
    const std::vector<int>& dexc,
    const int astates,
    const int bstates,
    const int ndexc,
    const Tensor& h1e,
    const int norbs,
    const bool is_alpha)
{
    const int states1 = is_alpha ? astates : bstates;
    const int states2 = is_alpha ? bstates : astates;
    const int inc1 = is_alpha ? bstates : 1;
    const int inc2 = is_alpha ? 1 : bstates;

    for (int s1 = 0; s1 < states1; ++s1) {
        const int* cdexc = dexc.data() + 3 * s1 * ndexc;
        const int* lim1 = cdexc + 3 * ndexc;
        std::complex<double>* cout = out.data().data() + s1 * inc1;

        for (; cdexc < lim1; cdexc = cdexc + 3) {
            const int target = cdexc[0];
            const int ijshift = cdexc[1];
            const int parity = cdexc[2];

            const std::complex<double> pref = static_cast<double>(parity) * h1e.read_data()[ijshift];
            const std::complex<double>* xptr = C_.data().data() + target * inc1;

            math_zaxpy(states2, pref, xptr, inc2, cout, inc2);
        }
    }
}


/// apply a 1-body and 2-body TensorOperator to the current state 
void apply_tensor_spin_12_body(const TensorOperator& top){
    // Stuff
}

void apply_sqop(const SQOperator& sqop){
    // Stuff
}

/// apply a constant to the FCI quantum computer.
void scale(const std::complex<double> a);


// std::vector<std::complex<double>> FCIComputer::direct_expectation_value(const TensorOperator& top){
//     // Stuff
// }

void FCIComputer::set_state(const Tensor other_state) {
    // Stuff
}

void FCIComputer::zero() {
    // Stuff
}

/// Sets all coefficeints fo the FCI Computer to Zero except the HF Determinant (set to 1).
void FCIComputer::hartree_fock() {
    C_.zero();
    C_.set({0, 0}, 1.0);
}



