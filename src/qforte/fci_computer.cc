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
// std::vector<double> apply_array_spin1(const std::vector<double>& h1e, int norb) {
// assert(h1e.size() == (norb * 2) * (norb * 2));

// int ncol = 0;
// int jorb = 0;
// for (int j = 0; j < norb * 2; ++j) {
// bool anyNonZero = false;
// for (int i = 0; i < norb * 2; ++i) {
//     if (h1e[i + j * (norb * 2)] != 0) {
//         anyNonZero = true;
//         break;
//     }
// }
// if (anyNonZero) {
//     ncol += 1;
//     jorb = j;
// }
// if (ncol > 1) {
//     break;
// }
// }

// std::vector<double> out;
// if (ncol > 1) {
// // Implementation of dense_apply_array_spin1_lm
// out = lm_apply_array1(coeff, {h1e.begin(), h1e.begin() + norb * norb},
//                         core._dexca, lena(), lenb(), norb, true);
// lm_apply_array1(coeff, {h1e.begin() + norb * norb, h1e.end()},
//                 core._dexcb, lena(), lenb(), norb, false, out);
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
// }

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



