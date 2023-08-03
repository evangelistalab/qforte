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

// #if defined(_OPENMP)
// #include <omp.h>
// extern const bool parallelism_enabled = true;
// #else
// extern const bool parallelism_enabled = false;
// #endif

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

    state_.zero_with_shape({nalfa_strs_, nbeta_strs_});
    state_.set_name("FCI Computer");
}

/// apply a TensorOperator to the current state 
void apply_tensor_operator(const TensorOperator& top);

/// apply a 1-body TensorOperator to the current state 
void apply_tensor_spin_1bdy(const TensorOperator& top);

// TODO(Nick): Uncomment, will need someting like the following:
std::vector<double> apply_tensor_spin_1bdy(const Tensor& h1e, size_t norb) {

    // if(h1e.size() != (norb * 2) * (norb * 2)){
    //     throw std::invalid_argument("Expecting h1e to be nso x nso for apply_tensor_spin_1bdy");
    // }

    // // Not sure what this is checking for?
    // size_t ncol = 0;
    // size_t jorb = 0;
    // for (size_t j = 0; j < norb * 2; ++j) {
    //     bool any_non_zero = false;
    //     for (int i = 0; i < norb * 2; ++i) {
    //         if (h1e.read_data()[i + j * (norb * 2)] != 0.0) {
    //             any_non_zero = true;
    //             break;
    //         }
    //     }
    //     if (any_non_zero) {
    //         ncol += 1;
    //         jorb = j;
    //     }
    //     if (ncol > 1) { break; }
    // }

    // std::vector<double> out;
    // if (ncol > 1) {
    //     // Implementation of dense_apply_array_spin1_lm
    //     out = lm_apply_array1(coeff, {h1e.begin(), h1e.begin() + norb * norb},
    //                             core._dexca, lena(), lenb(), norb, true);

    //     lm_apply_array1(coeff, {h1e.begin() + norb * norb, h1e.end()},
    //                     core._dexcb, lena(), lenb(), norb, false, out);
    // } else {
    // if (jorb < norb) {
    //     std::vector<double> dvec = calculate_dvec_spin_fixed_j(jorb);
    //     std::vector<double> h1eview(norb);
    //     for (int i = 0; i < norb; ++i) {
    //         h1eview[i] = h1e[i + jorb * (norb * 2)];
    //     }
    //     out = tensordot(h1eview, dvec, 1);
    // } else {
    //     std::vector<double> dvec = calculate_dvec_spin_fixed_j(jorb);
    //     std::vector<double> h1eview(norb);
    //     for (int i = 0; i < norb; ++i) {
    //         h1eview[i] = h1e[i + jorb * (norb * 2)];
    //     }
    //     out = tensordot(h1eview, dvec, 1);
    // }
    // }

    // return out;
}

void FCIComputer::lm_apply_array1_new(
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
        // const int* cdexc = &dexc[3 * s1 * ndexc];
        const int* cdexc = dexc.data() + 3 * s1 * ndexc;
        const int* lim1 = cdexc + 3 * ndexc;
        // std::complex<double>* cout = &out[s1 * inc1];
        std::complex<double>* cout = out.data().data() + s1 * inc1;
        for (; cdexc < lim1; cdexc = cdexc + 3) {
            const int target = cdexc[0];
            const int ijshift = cdexc[1];
            const int parity = cdexc[2];

            const std::complex<double> pref = static_cast<double>(parity) * h1e.read_data()[ijshift];
            // const std::complex<double>* xptr = &coeff[target * inc1];
            const std::complex<double>* xptr = state_.data().data() + target * inc1;
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
    // Stuff
}



