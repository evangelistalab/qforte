/// Basically just a continuation of the regular tensor class...

#include <tensor.h>
#include <blas_math.h>
#include <iostream>

#include <stdexcept>
#include <algorithm>
#include <map>

// namespace {

bool is_sequential(
    const std::vector<std::pair<int, std::string> >& vec)
{
    for (size_t ind = 1; ind < vec.size(); ind++) {
        if (vec[ind].first - vec[ind-1].first != 1) return false;
    }
    return true;
}

// }

// namespace lightspeed {

/// NICK: Plan, implement and sandbox-test 1. chain, 2. permute, 3, einsum...

/// chain requires
/// zscale
/// zaxpby
/// zgemm


Tensor Tensor::chain(
    const std::vector<Tensor >& As,
    const std::vector<bool>& trans,
    // const Tensor& C,
    std::complex<double> alpha,
    std::complex<double> beta)
{
    if (As.size() < 2) throw std::runtime_error("As must have at least two elements");
    if (As.size() != trans.size()) throw std::runtime_error("As and trans must be same size");
    for (size_t i = 0; i < As.size(); i++) {
        As[i].ndim_error(2);
    }

    for (size_t i = 1; i < As.size(); i++) {
        size_t nK1 = (trans[i-1] ? As[i-1].shape()[0] : As[i-1].shape()[1]);
        size_t nK2 = (trans[i] ? As[i].shape()[1] : As[i].shape()[0]);
        if (nK1 != nK2) throw std::runtime_error("Zip dimensions do not match.");
    }

    Tensor L = As[0];
    Tensor R = As[1];

    bool transL = trans[0];
    bool transR = trans[1];

    size_t nL = (transL ? L.shape()[1] : L.shape()[0]);
    size_t nR = (transR ? R.shape()[0] : R.shape()[1]);
    size_t nK = (transL ? L.shape()[0] : L.shape()[1]); 

    std::vector<size_t> LRdim;
    LRdim.push_back(nL);
    LRdim.push_back(nR);
    Tensor T(LRdim);

    math_zgemm(
        (transL ? 'T' : 'N'),
        (transR ? 'T' : 'N'),
        nL, 
        nR, 
        nK, 
        1.0,
        L.read_data().data(),
        (transL ? nL : nK),
        R.read_data().data(),
        (transR ? nK : nR),
        0.0,
        T.data().data(),
        nR);

    for (size_t i = 2; i < As.size(); i++) {
        L = T;
        //nL = nL;
        nK = nR;
        transL = false;
        R = As[i];
        transR = trans[i];
        nR = (transR ? R.shape()[0] : R.shape()[1]);
        std::vector<size_t> LR2dim;
        LR2dim.push_back(nL);
        LR2dim.push_back(nR);
        Tensor T(LR2dim);

        math_zgemm(
            (transL ? 'T' : 'N'),
            (transR ? 'T' : 'N'),
            nL, 
            nR, 
            nK, 
            1.0,
            L.read_data().data(),
            (transL ? nL : nK),
            R.read_data().data(),
            (transR ? nK : nR),
            0.0,
            T.data().data(),
            nR);
    }

    T.scale(alpha);
    return T;

    /// NICK: Can remove if we don't need to pass the C tensor...

    // if (!C) {
    //     // T->scale(alpha);
    //     T.scale(alpha);
    //     return T;
    // }

    // // C->shape_error(T->shape()); OLD
    // C.shape_error(T.shape());
    // // C->shape_error(T->shape()); OLD
    // C.shape_error(T.shape());
    // // C->axpby(T, alpha, beta);  OLD
    // C.zaxpby(T, alpha, beta, 1, 1); 

    // return C;
}




// std::shared_ptr<Tensor> Tensor::permute(
// Tensor Tensor::permute(
void Tensor::permute( /// NICK: will try replacement only model here
    const std::vector<std::string>& Ainds,
    const std::vector<std::string>& Cinds,
    const Tensor& A,
    // const Tensor& C2,
    Tensor& C2,
    std::complex<double> alpha,
    std::complex<double> beta)
{
    if (Ainds.size() != Cinds.size()) {
        throw std::runtime_error("permute: A and C inds must have same ndim");
    }

    // if (Ainds.size() != A->ndim()) { OLD
    if (Ainds.size() != A.ndim()) {
        throw std::runtime_error("permute: A tensor and inds must have same ndim");
    }

    // A_[...] = C_[perm(...)] 
    std::vector<int> perm;
    for (size_t ind = 0; ind < Ainds.size(); ind++) {
        const std::string& Aind = Ainds[ind];
        const std::vector<std::string>::const_iterator it = std::find(Cinds.begin(),Cinds.end(),Aind);
        if (it == Cinds.end()) {
            throw std::runtime_error("permute: A and C inds do not match");
        }
        int off = std::distance(Cinds.begin(),it);
        perm.push_back(off);
    }

    // Allocate C if needed
    // std::shared_ptr<Tensor> C = C2; OLD
    Tensor C = C2; // Shallow copy?

    /// NICK: Might be a todo,
    /// not sure when we would pass C2 as an empty tensor,
    /// but may need to uncomment the blow code if this is the case.
    if (!C.initialized()) {
        std::vector<size_t> Cdim(perm.size());
        for (size_t dim = 0; dim < perm.size(); dim++) {
            Cdim[perm[dim]] = A.shape()[dim];
        } 
        // C = std::shared_ptr<Tensor>(new Tensor(Cdim,"C"));
        // C.zero_with_shape(Cdim); // May want to mane this
    }
    
    // if (Cinds.size() != C->ndim()) { OLD
    if (Cinds.size() != C.ndim()) {
        throw std::runtime_error("permute: C tensor and inds must have same ndim");
    }
    for (size_t dim = 0; dim < perm.size(); dim++) {
        // if (A->shape()[dim] != C->shape()[perm[dim]]) { OLD
        if (A.shape()[dim] != C.shape()[perm[dim]]) {
            throw std::runtime_error("permute: A and C permuted inds must have same shape");
        }
    }

    // C->scale(beta);  OLD
    C.scale(beta);  
    
    int fast_dims = 0;
    size_t fast_size = 1L;
    for (int dim = ((int) perm.size()) - 1; dim >= 0; dim--) {
        if (dim == ((int) perm[dim])) {
            fast_dims++;
            // fast_size *= A->shape()[dim]; OLD
            fast_size *= A.shape()[dim];
        } else {
            break;
        }
    }

    int slow_dims = perm.size() - fast_dims;

    if (slow_dims == 0) {
        // std::complex<double>* Ap = A->data().data(); OLD
        const std::complex<double>* Ap = A.read_data().data();
        // std::complex<double>* Cp = C->data().data(); OLD
        std::complex<double>* Cp = C.data().data();
        // C_DAXPY(fast_size,alpha,Ap,1,Cp,1); OLD
        math_zaxpy(
            fast_size, 
            alpha,
            Ap,
            1,
            Cp,
            1);

        C2 = C;
        return;
    }

    // size_t slow_size = A->size() / fast_size;
    size_t slow_size = A.size() / fast_size;

    std::vector<size_t> CstridesA(slow_dims,0L);
    for (int dim = 0; dim < slow_dims; dim++) {
        // CstridesA[dim] = C->strides()[perm[dim]]; OLD
        CstridesA[dim] = C.strides()[perm[dim]];
    }

    // const std::vector<size_t>& Ashape = A->shape();
    const std::vector<size_t>& Ashape = A.shape();
    // const std::vector<size_t>& Astrides = A->strides();
    const std::vector<size_t>& Astrides = A.strides();
    
    // std::complex<double>* Ap = A->data().data(); OLD
    // std::complex<double>* Cp = C->data().data(); OLD

    const std::complex<double>* Ap = A.read_data().data();
    std::complex<double>* Cp = C.data().data();

    if (fast_size == 1L) {
        if (slow_dims == 2) {
            // #pragma omp parallel for
            for (size_t Aind0 = 0L; Aind0 < Ashape[0]; Aind0++) {
                const std::complex<double>* Atp = Ap + Aind0 * Astrides[0];
            for (size_t Aind1 = 0L; Aind1 < Ashape[1]; Aind1++) {
                std::complex<double>* Ctp = Cp +
                    Aind0 * CstridesA[0] +
                    Aind1 * CstridesA[1];
                (*Ctp) += alpha * (*Atp);
                Atp += fast_size;
            }}
        } else if (slow_dims == 3) {
            // #pragma omp parallel for
            for (size_t Aind0 = 0L; Aind0 < Ashape[0]; Aind0++) {
                const std::complex<double>* Atp = Ap + Aind0 * Astrides[0];
            for (size_t Aind1 = 0L; Aind1 < Ashape[1]; Aind1++) {
            for (size_t Aind2 = 0L; Aind2 < Ashape[2]; Aind2++) {
                std::complex<double>* Ctp = Cp +
                    Aind0 * CstridesA[0] +
                    Aind1 * CstridesA[1] +
                    Aind2 * CstridesA[2];
                (*Ctp) += alpha * (*Atp);
                Atp += fast_size;
            }}}
        } else if (slow_dims == 4) {
            // #pragma omp parallel for
            for (size_t Aind0 = 0L; Aind0 < Ashape[0]; Aind0++) {
                const std::complex<double>* Atp = Ap + Aind0 * Astrides[0];
            for (size_t Aind1 = 0L; Aind1 < Ashape[1]; Aind1++) {
            for (size_t Aind2 = 0L; Aind2 < Ashape[2]; Aind2++) {
            for (size_t Aind3 = 0L; Aind3 < Ashape[3]; Aind3++) {
                std::complex<double>* Ctp = Cp +
                    Aind0 * CstridesA[0] +
                    Aind1 * CstridesA[1] +
                    Aind2 * CstridesA[2] +
                    Aind3 * CstridesA[3];
                (*Ctp) += alpha * (*Atp);
                Atp += fast_size;
            }}}}
        } else if (slow_dims == 5) {
            // #pragma omp parallel for
            for (size_t Aind0 = 0L; Aind0 < Ashape[0]; Aind0++) {
                const std::complex<double>* Atp = Ap + Aind0 * Astrides[0];
            for (size_t Aind1 = 0L; Aind1 < Ashape[1]; Aind1++) {
            for (size_t Aind2 = 0L; Aind2 < Ashape[2]; Aind2++) {
            for (size_t Aind3 = 0L; Aind3 < Ashape[3]; Aind3++) {
            for (size_t Aind4 = 0L; Aind4 < Ashape[4]; Aind4++) {
                std::complex<double>* Ctp = Cp +
                    Aind0 * CstridesA[0] +
                    Aind1 * CstridesA[1] +
                    Aind2 * CstridesA[2] +
                    Aind3 * CstridesA[3] +
                    Aind4 * CstridesA[4];
                (*Ctp) += alpha * (*Atp);
                Atp += fast_size;
            }}}}}
        } else if (slow_dims == 6) {
            // #pragma omp parallel for
            for (size_t Aind0 = 0L; Aind0 < Ashape[0]; Aind0++) {
                const std::complex<double>* Atp = Ap + Aind0 * Astrides[0];
            for (size_t Aind1 = 0L; Aind1 < Ashape[1]; Aind1++) {
            for (size_t Aind2 = 0L; Aind2 < Ashape[2]; Aind2++) {
            for (size_t Aind3 = 0L; Aind3 < Ashape[3]; Aind3++) {
            for (size_t Aind4 = 0L; Aind4 < Ashape[4]; Aind4++) {
            for (size_t Aind5 = 0L; Aind5 < Ashape[5]; Aind5++) {
                std::complex<double>* Ctp = Cp +
                    Aind0 * CstridesA[0] +
                    Aind1 * CstridesA[1] +
                    Aind2 * CstridesA[2] +
                    Aind3 * CstridesA[3] +
                    Aind4 * CstridesA[4] +
                    Aind5 * CstridesA[5];
                (*Ctp) += alpha * (*Atp);
                Atp += fast_size;
            }}}}}}
        } else if (slow_dims == 7) {
            // #pragma omp parallel for
            for (size_t Aind0 = 0L; Aind0 < Ashape[0]; Aind0++) {
                const std::complex<double>* Atp = Ap + Aind0 * Astrides[0];
            for (size_t Aind1 = 0L; Aind1 < Ashape[1]; Aind1++) {
            for (size_t Aind2 = 0L; Aind2 < Ashape[2]; Aind2++) {
            for (size_t Aind3 = 0L; Aind3 < Ashape[3]; Aind3++) {
            for (size_t Aind4 = 0L; Aind4 < Ashape[4]; Aind4++) {
            for (size_t Aind5 = 0L; Aind5 < Ashape[5]; Aind5++) {
            for (size_t Aind6 = 0L; Aind6 < Ashape[6]; Aind6++) {
                std::complex<double>* Ctp = Cp +
                    Aind0 * CstridesA[0] +
                    Aind1 * CstridesA[1] +
                    Aind2 * CstridesA[2] +
                    Aind3 * CstridesA[3] +
                    Aind4 * CstridesA[4] +
                    Aind5 * CstridesA[5] +
                    Aind6 * CstridesA[6];
                (*Ctp) += alpha * (*Atp);
                Atp += fast_size;
            }}}}}}}
        } else if (slow_dims == 8) {
            // #pragma omp parallel for
            for (size_t Aind0 = 0L; Aind0 < Ashape[0]; Aind0++) {
                const std::complex<double>* Atp = Ap + Aind0 * Astrides[0];
            for (size_t Aind1 = 0L; Aind1 < Ashape[1]; Aind1++) {
            for (size_t Aind2 = 0L; Aind2 < Ashape[2]; Aind2++) {
            for (size_t Aind3 = 0L; Aind3 < Ashape[3]; Aind3++) {
            for (size_t Aind4 = 0L; Aind4 < Ashape[4]; Aind4++) {
            for (size_t Aind5 = 0L; Aind5 < Ashape[5]; Aind5++) {
            for (size_t Aind6 = 0L; Aind6 < Ashape[6]; Aind6++) {
            for (size_t Aind7 = 0L; Aind7 < Ashape[7]; Aind7++) {
                std::complex<double>* Ctp = Cp +
                    Aind0 * CstridesA[0] +
                    Aind1 * CstridesA[1] +
                    Aind2 * CstridesA[2] +
                    Aind3 * CstridesA[3] +
                    Aind4 * CstridesA[4] +
                    Aind5 * CstridesA[5] +
                    Aind6 * CstridesA[6] +
                    Aind6 * CstridesA[7];
                (*Ctp) += alpha * (*Atp);
                Atp += fast_size;
            }}}}}}}}
        } else {
            // #pragma omp parallel for
            for (size_t ind = 0L; ind < slow_size; ind++) {
                const std::complex<double>* Atp = Ap + ind * fast_size;
                std::complex<double>* Ctp = Cp;
                size_t num = ind;
                for (int dim = slow_dims - 1; dim >= 0; dim--) {
                    size_t val = num % Ashape[dim];
                    num /= Ashape[dim];
                    Ctp += val * CstridesA[dim];
                }
                (*Ctp) += alpha * (*Atp); 
            }
        }
    } else {
        if (slow_dims == 2) {
            // #pragma omp parallel for
            for (size_t Aind0 = 0L; Aind0 < Ashape[0]; Aind0++) {
                const std::complex<double>* Atp = Ap + Aind0 * Astrides[0];
            for (size_t Aind1 = 0L; Aind1 < Ashape[1]; Aind1++) {
                std::complex<double>* Ctp = Cp +
                    Aind0 * CstridesA[0] +
                    Aind1 * CstridesA[1];
                // C_DAXPY(fast_size,alpha,Atp,1,Ctp,1); OLD
                math_zaxpy(
                    fast_size,
                    alpha,
                    Atp,
                    1,
                    Ctp,
                    1);
                Atp += fast_size;
            }}
        } else if (slow_dims == 3) {
            // #pragma omp parallel for
            for (size_t Aind0 = 0L; Aind0 < Ashape[0]; Aind0++) {
                const std::complex<double>* Atp = Ap + Aind0 * Astrides[0];
            for (size_t Aind1 = 0L; Aind1 < Ashape[1]; Aind1++) {
            for (size_t Aind2 = 0L; Aind2 < Ashape[2]; Aind2++) {
                std::complex<double>* Ctp = Cp +
                    Aind0 * CstridesA[0] +
                    Aind1 * CstridesA[1] +
                    Aind2 * CstridesA[2];
                // C_DAXPY(fast_size,alpha,Atp,1,Ctp,1); OLD
                math_zaxpy(
                    fast_size,
                    alpha,
                    Atp,
                    1,
                    Ctp,
                    1);
                Atp += fast_size;
            }}}
        } else if (slow_dims == 4) {
            // #pragma omp parallel for
            for (size_t Aind0 = 0L; Aind0 < Ashape[0]; Aind0++) {
                const std::complex<double>* Atp = Ap + Aind0 * Astrides[0];
            for (size_t Aind1 = 0L; Aind1 < Ashape[1]; Aind1++) {
            for (size_t Aind2 = 0L; Aind2 < Ashape[2]; Aind2++) {
            for (size_t Aind3 = 0L; Aind3 < Ashape[3]; Aind3++) {
                std::complex<double>* Ctp = Cp +
                    Aind0 * CstridesA[0] +
                    Aind1 * CstridesA[1] +
                    Aind2 * CstridesA[2] +
                    Aind3 * CstridesA[3];
                // C_DAXPY(fast_size,alpha,Atp,1,Ctp,1);
                math_zaxpy(
                    fast_size,
                    alpha,
                    Atp,
                    1,
                    Ctp,
                    1);
                Atp += fast_size;
            }}}}
        } else if (slow_dims == 5) {
            // #pragma omp parallel for
            for (size_t Aind0 = 0L; Aind0 < Ashape[0]; Aind0++) {
                const std::complex<double>* Atp = Ap + Aind0 * Astrides[0];
            for (size_t Aind1 = 0L; Aind1 < Ashape[1]; Aind1++) {
            for (size_t Aind2 = 0L; Aind2 < Ashape[2]; Aind2++) {
            for (size_t Aind3 = 0L; Aind3 < Ashape[3]; Aind3++) {
            for (size_t Aind4 = 0L; Aind4 < Ashape[4]; Aind4++) {
                std::complex<double>* Ctp = Cp +
                    Aind0 * CstridesA[0] +
                    Aind1 * CstridesA[1] +
                    Aind2 * CstridesA[2] +
                    Aind3 * CstridesA[3] +
                    Aind4 * CstridesA[4];
                // C_DAXPY(fast_size,alpha,Atp,1,Ctp,1); OLD
                math_zaxpy(
                    fast_size,
                    alpha,
                    Atp,
                    1,
                    Ctp,
                    1);
                Atp += fast_size;
            }}}}}
        } else if (slow_dims == 6) {
            // #pragma omp parallel for
            for (size_t Aind0 = 0L; Aind0 < Ashape[0]; Aind0++) {
                const std::complex<double>* Atp = Ap + Aind0 * Astrides[0];
            for (size_t Aind1 = 0L; Aind1 < Ashape[1]; Aind1++) {
            for (size_t Aind2 = 0L; Aind2 < Ashape[2]; Aind2++) {
            for (size_t Aind3 = 0L; Aind3 < Ashape[3]; Aind3++) {
            for (size_t Aind4 = 0L; Aind4 < Ashape[4]; Aind4++) {
            for (size_t Aind5 = 0L; Aind5 < Ashape[5]; Aind5++) {
                std::complex<double>* Ctp = Cp +
                    Aind0 * CstridesA[0] +
                    Aind1 * CstridesA[1] +
                    Aind2 * CstridesA[2] +
                    Aind3 * CstridesA[3] +
                    Aind4 * CstridesA[4] +
                    Aind5 * CstridesA[5];
                // C_DAXPY(fast_size,alpha,Atp,1,Ctp,1); OLD
                math_zaxpy(
                    fast_size,
                    alpha,
                    Atp,
                    1,
                    Ctp,
                    1);
                Atp += fast_size;
            }}}}}}
        } else if (slow_dims == 7) {
            // #pragma omp parallel for
            for (size_t Aind0 = 0L; Aind0 < Ashape[0]; Aind0++) {
                const std::complex<double>* Atp = Ap + Aind0 * Astrides[0];
            for (size_t Aind1 = 0L; Aind1 < Ashape[1]; Aind1++) {
            for (size_t Aind2 = 0L; Aind2 < Ashape[2]; Aind2++) {
            for (size_t Aind3 = 0L; Aind3 < Ashape[3]; Aind3++) {
            for (size_t Aind4 = 0L; Aind4 < Ashape[4]; Aind4++) {
            for (size_t Aind5 = 0L; Aind5 < Ashape[5]; Aind5++) {
            for (size_t Aind6 = 0L; Aind6 < Ashape[6]; Aind6++) {
                std::complex<double>* Ctp = Cp +
                    Aind0 * CstridesA[0] +
                    Aind1 * CstridesA[1] +
                    Aind2 * CstridesA[2] +
                    Aind3 * CstridesA[3] +
                    Aind4 * CstridesA[4] +
                    Aind5 * CstridesA[5] +
                    Aind6 * CstridesA[6];
                // C_DAXPY(fast_size,alpha,Atp,1,Ctp,1); OLD
                math_zaxpy(
                    fast_size,
                    alpha,
                    Atp,
                    1,
                    Ctp,
                    1);
                Atp += fast_size;
            }}}}}}}
        } else if (slow_dims == 8) {
            // #pragma omp parallel for
            for (size_t Aind0 = 0L; Aind0 < Ashape[0]; Aind0++) {
                const std::complex<double>* Atp = Ap + Aind0 * Astrides[0];
            for (size_t Aind1 = 0L; Aind1 < Ashape[1]; Aind1++) {
            for (size_t Aind2 = 0L; Aind2 < Ashape[2]; Aind2++) {
            for (size_t Aind3 = 0L; Aind3 < Ashape[3]; Aind3++) {
            for (size_t Aind4 = 0L; Aind4 < Ashape[4]; Aind4++) {
            for (size_t Aind5 = 0L; Aind5 < Ashape[5]; Aind5++) {
            for (size_t Aind6 = 0L; Aind6 < Ashape[6]; Aind6++) {
            for (size_t Aind7 = 0L; Aind7 < Ashape[7]; Aind7++) {
                std::complex<double>* Ctp = Cp +
                    Aind0 * CstridesA[0] +
                    Aind1 * CstridesA[1] +
                    Aind2 * CstridesA[2] +
                    Aind3 * CstridesA[3] +
                    Aind4 * CstridesA[4] +
                    Aind5 * CstridesA[5] +
                    Aind6 * CstridesA[6] +
                    Aind6 * CstridesA[7];
                // C_DAXPY(fast_size,alpha,Atp,1,Ctp,1);
                math_zaxpy(
                    fast_size,
                    alpha,
                    Atp,
                    1,
                    Ctp,
                    1);
                Atp += fast_size;
            }}}}}}}}
        } else {
            // #pragma omp parallel for
            for (size_t ind = 0L; ind < slow_size; ind++) {
                const std::complex<double>* Atp = Ap + ind * fast_size;
                std::complex<double>* Ctp = Cp;
                size_t num = ind;
                for (int dim = slow_dims - 1; dim >= 0; dim--) {
                    size_t val = num % Ashape[dim];
                    num /= Ashape[dim];
                    Ctp += val * CstridesA[dim];
                }
                // C_DAXPY(fast_size,alpha,Atp,1,Ctp,1); 
                math_zaxpy(
                    fast_size,
                    alpha,
                    Atp,
                    1,
                    Ctp,
                    1); 
            }
        }
    }
    // return C;

    /// NICK: This seems to work but is obviously super wasteful, 
    /// May want to revise.
    C2 = C;
    return;
} // Well, it compiles at least!



void Tensor::einsum(
    const std::vector<std::string>& Ainds,
    const std::vector<std::string>& Binds,
    const std::vector<std::string>& Cinds,
    const Tensor& A,
    const Tensor& B,
    // const Tensor& C3, 
    Tensor& C3, 
    std::complex<double> alpha,
    std::complex<double> beta)
{
    // ==> Indexing Logic <== //

    // if (Ainds.size() != A->ndim()) { OLD
    if (Ainds.size() != A.ndim()) {
        throw std::runtime_error("einsum: A tensor and inds must have same ndim");
    }
    // if (Binds.size() != B->ndim()) {
    if (Binds.size() != B.ndim()) {
        throw std::runtime_error("einsum: B tensor and inds must have same ndim");
    }
    
    /// Allocate C if needed
    // std::shared_ptr<Tensor> C = C3;
    Tensor C = C3; /// IS A DEEP COPY, should be a shallow copy...

    // std::cout << " C.get({0}):  " << C.get({0}) << std::endl;

    // C3.set({0}, 1.5632);

    // std::cout << " C.get({0}):  " <<  C.get({0}) << std::endl;
    // std::cout << "C3.get({0}):  " << C3.get({0}) << std::endl;

    if (!C.initialized()) {
        std::vector<size_t> Cdims(Cinds.size());
        for (size_t Crank = 0; Crank < Cinds.size(); Crank++) {
            bool found = false;
            if (!found) {
                for (size_t Arank = 0; Arank < Ainds.size(); Arank++) {
                    if (Ainds[Arank] == Cinds[Crank]) {
                        // Cdims[Crank] = A->shape()[Arank];  
                        Cdims[Crank] = A.shape()[Arank];  
                        found = true;
                        break;
                    }
                }
            }
            if (!found) {
                for (size_t Brank = 0; Brank < Binds.size(); Brank++) {
                    if (Binds[Brank] == Cinds[Crank]) {
                        // Cdims[Crank] = B->shape()[Brank];  
                        Cdims[Crank] = B.shape()[Brank];  
                        found = true;
                        break;
                    }
                }
            }
            if (!found) throw std::runtime_error("einsum: Cind not present in Ainds or Binds: " + Cinds[Crank]);
        }
        // C = std::shared_ptr<Tensor>(new Tensor(Cdims, "C"));
        // C.zero_with_shape(Cdims);
    }

    // if (Cinds.size() != C->ndim()) {
    if (Cinds.size() != C.ndim()) {
        throw std::runtime_error("einsum: C tensor and inds must have same ndim");
    }

    // Unique inds
    std::vector<std::string> inds;
    inds.insert(inds.end(),Ainds.begin(),Ainds.end()); 
    inds.insert(inds.end(),Binds.begin(),Binds.end()); 
    inds.insert(inds.end(),Cinds.begin(),Cinds.end()); 
    std::sort(inds.begin(), inds.end());
    inds.resize(std::distance(inds.begin(), std::unique(inds.begin(),inds.end())));

    // Determine index types and compound index sizes
    size_t Psize = 1L;
    size_t isize = 1L;
    size_t jsize = 1L;
    size_t ksize = 1L;

    std::vector<std::string> compound_names;
    compound_names.push_back("PC");
    compound_names.push_back("PA");
    compound_names.push_back("PB");
    compound_names.push_back("iC");
    compound_names.push_back("iA");
    compound_names.push_back("jC");
    compound_names.push_back("jB");
    compound_names.push_back("kA");
    compound_names.push_back("kB");

    // For each compound name, { < A/B/Cpos, name > }
    std::map<std::string, std::vector<std::pair<int, std::string> > > compound_inds;
    compound_inds["PC"] = std::vector<std::pair<int, std::string> >();
    compound_inds["PA"] = std::vector<std::pair<int, std::string> >();
    compound_inds["PB"] = std::vector<std::pair<int, std::string> >();
    compound_inds["iC"] = std::vector<std::pair<int, std::string> >();
    compound_inds["iA"] = std::vector<std::pair<int, std::string> >();
    compound_inds["jC"] = std::vector<std::pair<int, std::string> >();
    compound_inds["jB"] = std::vector<std::pair<int, std::string> >();
    compound_inds["kA"] = std::vector<std::pair<int, std::string> >();
    compound_inds["kB"] = std::vector<std::pair<int, std::string> >();
    for (size_t ind2 = 0; ind2 < inds.size(); ind2++) {
        const std::string& ind = inds[ind2];
        size_t Apos = std::distance(Ainds.begin(),std::find(Ainds.begin(),Ainds.end(),ind));
        size_t Bpos = std::distance(Binds.begin(),std::find(Binds.begin(),Binds.end(),ind));
        size_t Cpos = std::distance(Cinds.begin(),std::find(Cinds.begin(),Cinds.end(),ind));
        if (Apos != Ainds.size() && Bpos != Binds.size() && Cpos != Cinds.size()) {
            // if (C->shape()[Cpos] != A->shape()[Apos] || C->shape()[Cpos] != B->shape()[Bpos]) {
            if (C.shape()[Cpos] != A.shape()[Apos] || C.shape()[Cpos] != B.shape()[Bpos]) {
                throw std::runtime_error("einsum: Invalid P (Hadamard) index: " + ind);
            }
            compound_inds["PC"].push_back(std::make_pair(Cpos, ind));
            compound_inds["PA"].push_back(std::make_pair(Apos, ind));
            compound_inds["PB"].push_back(std::make_pair(Bpos, ind));
            // Psize *= C->shape()[Cpos];
            Psize *= C.shape()[Cpos];
        } else if (Cpos != Cinds.size() && Apos != Ainds.size()) {
            // if (C->shape()[Cpos] != A->shape()[Apos]) {
            if (C.shape()[Cpos] != A.shape()[Apos]) {
                throw std::runtime_error("einsum: Invalid i (Left) index: " + ind);
            }
            compound_inds["iC"].push_back(std::make_pair(Cpos, ind));
            compound_inds["iA"].push_back(std::make_pair(Apos, ind));
            // isize *= C->shape()[Cpos];
            isize *= C.shape()[Cpos];
        } else if (Cpos != Cinds.size() && Bpos != Binds.size()) {
            // if (C->shape()[Cpos] != B->shape()[Bpos]) {
            if (C.shape()[Cpos] != B.shape()[Bpos]) {
                throw std::runtime_error("einsum: Invalid j (Right) index: " + ind);
            }
            compound_inds["jC"].push_back(std::make_pair(Cpos, ind));
            compound_inds["jB"].push_back(std::make_pair(Bpos, ind));
            // jsize *= C->shape()[Cpos];
            jsize *= C.shape()[Cpos];
        } else if (Apos != Ainds.size() && Bpos != Binds.size()) {
            // if (A->shape()[Apos] != B->shape()[Bpos]) {
            if (A.shape()[Apos] != B.shape()[Bpos]) {
                throw std::runtime_error("einsum: Invalid k (Contraction) index: " + ind);
            }
            compound_inds["kA"].push_back(std::make_pair(Apos, ind));
            compound_inds["kB"].push_back(std::make_pair(Bpos, ind));
            // ksize *= A->shape()[Apos];
            ksize *= A.shape()[Apos];
        } else {
            throw std::runtime_error("einsum: Index appears only once: " + ind);
        }
    }

    // Sort to place in absolute order encountered in each tensors
    std::sort(compound_inds["PC"].begin(),compound_inds["PC"].end());
    std::sort(compound_inds["PA"].begin(),compound_inds["PA"].end());
    std::sort(compound_inds["PB"].begin(),compound_inds["PB"].end());
    std::sort(compound_inds["iC"].begin(),compound_inds["iC"].end());
    std::sort(compound_inds["iA"].begin(),compound_inds["iA"].end());
    std::sort(compound_inds["jC"].begin(),compound_inds["jC"].end());
    std::sort(compound_inds["jB"].begin(),compound_inds["jB"].end());
    std::sort(compound_inds["kA"].begin(),compound_inds["kA"].end());
    std::sort(compound_inds["kB"].begin(),compound_inds["kB"].end());

    // Flags to mark if tensors must be permuted
    bool Aperm = false;
    bool Bperm = false;
    bool Cperm = false;

    // Permutation is required if composite indices are not sequential
    Cperm = Cperm || !is_sequential(compound_inds["PC"]);
    Cperm = Cperm || !is_sequential(compound_inds["iC"]);
    Cperm = Cperm || !is_sequential(compound_inds["jC"]);
    Aperm = Aperm || !is_sequential(compound_inds["PA"]);
    Aperm = Aperm || !is_sequential(compound_inds["iA"]);
    Aperm = Aperm || !is_sequential(compound_inds["kA"]);
    Bperm = Bperm || !is_sequential(compound_inds["PB"]);
    Bperm = Bperm || !is_sequential(compound_inds["jB"]);
    Bperm = Bperm || !is_sequential(compound_inds["kB"]);
    
    // Permutation is required if Hadamard indices are not slowest
    if (compound_inds["PC"].size()) {
        Aperm = Aperm || (compound_inds["PA"][0].first != 0);
        Bperm = Bperm || (compound_inds["PB"][0].first != 0);
        Cperm = Cperm || (compound_inds["PC"][0].first != 0);
    } 

    // Figure out transposes assuming no permutation (will fix later if permuted)
    bool Atrans = false;
    bool Btrans = false;
    bool Ctrans = false;
    if (compound_inds["iC"].size() && compound_inds["iC"][0].first != (int) compound_inds["PC"].size()) {
        Ctrans = true;
    }
    if (compound_inds["iA"].size() && compound_inds["iA"][0].first != (int) compound_inds["PC"].size()) {
        Atrans = true;
    }
    if (compound_inds["jB"].size() && compound_inds["jB"][0].first == (int) compound_inds["PC"].size()) {
        Btrans = true;
    }

    // Possible contiguous index orders (selection to be made later)
    std::map<std::string, std::vector<std::string> > compound_inds2;
    for (size_t ind = 0; ind < compound_names.size(); ind++) {
        const std::string& name = compound_names[ind];
        std::vector<std::string> vec;
        for (size_t ind2 = 0; ind2 < compound_inds[name].size(); ind2++) {
            vec.push_back(compound_inds[name][ind2].second);    
        }
        compound_inds2[name] = vec;
    }

    /**
    * Fix permutation order considerations
    *
    * Rules if a permutation mismatch is detected:
    * -If both tensors are already on the permute list, it doesn't matter which
    *is fixed
    * -Else if one tensor is already on the permute list but not the other, fix
    *the one that is already on the permute list
    * -Else fix the smaller tensor
    *
    * Note: this scheme is not optimal is permutation mismatches exist in P -
    *for reasons of simplicity, A and B are
    * permuted to C's P order, with no present considerations of better pathways
    **/
    if (compound_inds2["iC"] != compound_inds2["iA"]) {
        if (Cperm) {
            compound_inds2["iC"] = compound_inds2["iA"];
        } else if (Aperm) {
            compound_inds2["iA"] = compound_inds2["iC"];
        // } else if (C->size() <= A->size()) {
        } else if (C.size() <= A.size()) {
            compound_inds2["iC"] = compound_inds2["iA"];
            Cperm = true;
        } else {
            compound_inds2["iA"] = compound_inds2["iC"];
            Aperm = true;
        }
    }
    if (compound_inds2["jC"] != compound_inds2["jB"]) {
        if (Cperm) {
            compound_inds2["jC"] = compound_inds2["jB"];
        } else if (Bperm) {
            compound_inds2["jB"] = compound_inds2["jC"];
        // } else if (C->size() <= B->size()) {
        } else if (C.size() <= B.size()) {
            compound_inds2["jC"] = compound_inds2["jB"];
            Cperm = true;
        } else {
            compound_inds2["jB"] = compound_inds2["jC"];
            Bperm = true;
        }
    }
    if (compound_inds2["kA"] != compound_inds2["kB"]) {
        if (Aperm) {
            compound_inds2["kA"] = compound_inds2["kB"];
        } else if (Bperm) {
            compound_inds2["kB"] = compound_inds2["kA"];
        // } else if (A->size() <= B->size()) {
        } else if (A.size() <= B.size()) {
            compound_inds2["kA"] = compound_inds2["kB"];
            Aperm = true;
        } else {
            compound_inds2["kB"] = compound_inds2["kA"];
            Bperm = true;
        }
    }
    if (compound_inds2["PC"] != compound_inds2["PA"]) {
        compound_inds2["PA"] = compound_inds2["PC"];
        Aperm = true;
    }
    if (compound_inds2["PC"] != compound_inds2["PB"]) {
        compound_inds2["PB"] = compound_inds2["PC"];
        Bperm = true;
    }

    // Assign the permuted indices (if flagged for permute) or the original indices
    std::vector<std::string> Cinds2;
    std::vector<std::string> Ainds2;
    std::vector<std::string> Binds2;
    if (Cperm) {
        Cinds2.insert(Cinds2.end(), compound_inds2["PC"].begin(), compound_inds2["PC"].end());
        Cinds2.insert(Cinds2.end(), compound_inds2["iC"].begin(), compound_inds2["iC"].end());
        Cinds2.insert(Cinds2.end(), compound_inds2["jC"].begin(), compound_inds2["jC"].end());
        Ctrans = false;
    } else {
        Cinds2 = Cinds;
    }
    if (Aperm) {
        Ainds2.insert(Ainds2.end(), compound_inds2["PA"].begin(), compound_inds2["PA"].end());
        Ainds2.insert(Ainds2.end(), compound_inds2["iA"].begin(), compound_inds2["iA"].end());
        Ainds2.insert(Ainds2.end(), compound_inds2["kA"].begin(), compound_inds2["kA"].end());
        Atrans = false;
        // Atrans = true; // ?? Nick Test
    } else {
        Ainds2 = Ainds;
    }
    if (Bperm) {
        Binds2.insert(Binds2.end(), compound_inds2["PB"].begin(), compound_inds2["PB"].end());
        Binds2.insert(Binds2.end(), compound_inds2["jB"].begin(), compound_inds2["jB"].end());
        Binds2.insert(Binds2.end(), compound_inds2["kB"].begin(), compound_inds2["kB"].end());
        Btrans = true;
        // Btrans = false; // ?? Nick Test
    } else {
        Binds2 = Binds;
    }

    // So what exactly happened?
    // #if 0
    printf("==> Einsum Trace <==\n\n");
    printf("Original: C[");
    for (size_t ind = 0l; ind < Cinds.size(); ind++) {
        printf("%s", Cinds[ind].c_str());
    }
    printf("] = A[");
    for (size_t ind = 0l; ind < Ainds.size(); ind++) {
        printf("%s", Ainds[ind].c_str());
    }
    printf("] * B[");
    for (size_t ind = 0l; ind < Binds.size(); ind++) {
        printf("%s", Binds[ind].c_str());
    }
    printf("]\n");
    printf("New:      C[");
    for (size_t ind = 0l; ind < Cinds2.size(); ind++) {
        printf("%s", Cinds2[ind].c_str());
    }
    printf("] = A[");
    for (size_t ind = 0l; ind < Ainds2.size(); ind++) {
        printf("%s", Ainds2[ind].c_str());
    }
    printf("] * B[");
    for (size_t ind = 0l; ind < Binds2.size(); ind++) {
        printf("%s", Binds2[ind].c_str());
    }
    printf("]\n");
    printf("C Permuted: %s\n", Cperm ? "Yes" : "No");
    printf("A Permuted: %s\n", Aperm ? "Yes" : "No");
    printf("B Permuted: %s\n", Bperm ? "Yes" : "No");
    // printf("\n");
    
    // if(Cperm and Ctrans){ Ctrans = false;}
    // if(Aperm and Atrans){ Atrans = false;}
    // if(Bperm and Btrans){ Btrans = false;}

    printf("C Transposed: %s\n", Ctrans ? "Yes" : "No");
    printf("A Transposed: %s\n", Atrans ? "Yes" : "No");
    printf("B Transposed: %s\n", Btrans ? "Yes" : "No");
    printf("\n");
    // #endif

    // TODO: Do not permute if DGEMV or lower - use for loops

    // ==> Tensor Operations <== //

    // => Temporary Allocation/Permutation <= //
    
    /// NICK: This next section above the blas calls is a bit odd to me,
    /// may be why all of the tensors were kepts as shared pointer to tensors?

    // std::shared_ptr<Tensor> A2 = A;
    Tensor A2 = A;
    // std::shared_ptr<Tensor> B2 = B;
    Tensor B2 = B;
    // std::shared_ptr<Tensor> C2 = C;
    Tensor C2 = C;

    // std::cout << "\n mid: C  \n" << C.str() << std::endl;
    // std::cout << "\n mid: C2 \n" << C2.str() << std::endl;
    // std::cout << "\n mid: C3 \n" << C3.str() << std::endl;
    // std::cout << "\n mid: A2 \n" << A2.str() << std::endl;
    // std::cout << "\n mid: B2 \n" << B2.str() << std::endl;

    std::cout << "\n Psize: \n" << Psize << std::endl;
    

    if (Aperm) {
        std::vector<size_t> A2shape;
        for (size_t ind2 = 0; ind2 < Ainds2.size(); ind2++) {
            const std::string& ind = Ainds2[ind2];
            // A2shape.push_back(A->shape()[std::distance(Ainds.begin(),std::find(Ainds.begin(),Ainds.end(),ind))]);
            A2shape.push_back(A.shape()[std::distance(Ainds.begin(),std::find(Ainds.begin(),Ainds.end(),ind))]);
        } 
        // A2 = std::shared_ptr<Tensor>(new Tensor(A2shape));
        Tensor A2(A2shape);
        // Tensor::permute(Ainds, Ainds2, A, A2);
        // A2 = Tensor::permute(Ainds, Ainds2, A, A2); //Maybe?? Permute doesn't have to return someting?? A and A2 are const??
        Tensor::permute(Ainds, Ainds2, A, A2); //Maybe?? Permute doesn't have to return someting?? A and A2 are const??
    }
    if (Bperm) {
        std::vector<size_t> B2shape;
        for (size_t ind2 = 0; ind2 < Binds2.size(); ind2++) {
            const std::string& ind = Binds2[ind2];
            // B2shape.push_back(B->shape()[std::distance(Binds.begin(),std::find(Binds.begin(),Binds.end(),ind))]);
            B2shape.push_back(B.shape()[std::distance(Binds.begin(),std::find(Binds.begin(),Binds.end(),ind))]);
        } 
        // B2 = std::shared_ptr<Tensor>(new Tensor(B2shape));
        Tensor B2(B2shape);
        // Tensor::permute(Binds,Binds2,B,B2);
        // B2 = Tensor::permute(Binds, Binds2, B, B2);
        Tensor::permute(Binds, Binds2, B, B2);
    }
    if (Cperm) {
        std::vector<size_t> C2shape;
        for (size_t ind2 = 0; ind2 < Cinds2.size(); ind2++) {
            const std::string& ind = Cinds2[ind2];
            // C2shape.push_back(C->shape()[std::distance(Cinds.begin(),std::find(Cinds.begin(),Cinds.end(),ind))]);
            C2shape.push_back(C.shape()[std::distance(Cinds.begin(),std::find(Cinds.begin(),Cinds.end(),ind))]);
        } 
        // C2 = std::shared_ptr<Tensor>(new Tensor(C2shape));
        Tensor C2(C2shape);
        // Tensor::permute(Cinds,Cinds2,C,C2);
        // C2 = Tensor::permute(Cinds, Cinds2, C,  C2);
        Tensor::permute(Cinds, Cinds2, C,  C2);
    }
    
    // => BLAS <= //

    // std::complex<double>* A2p = A2->data().data();
    std::complex<double>* A2p = A2.data().data();
    // std::complex<double>* B2p = B2->data().data();
    std::complex<double>* B2p = B2.data().data();
    // std::complex<double>* C2p = C2->data().data();
    std::complex<double>* C2p = C2.data().data();

    for (size_t P = 0; P < Psize; P++) {
        char Ltrans;
        char Rtrans;
        size_t nrow;
        size_t ncol;
        std::complex<double>* Lp;
        std::complex<double>* Rp;
        size_t Llda;
        size_t Rlda;
        if (Ctrans) {
            Lp = B2p;
            Rp = A2p;
            nrow = jsize;
            ncol = isize;
            Ltrans = (Btrans ? 'N' : 'T');
            Rtrans = (Atrans ? 'N' : 'T');
            Llda = (Btrans ? ksize : jsize); 
            Rlda = (Atrans ? isize : ksize); 
        } else {
            Lp = A2p;
            Rp = B2p;
            nrow = isize;
            ncol = jsize;
            Ltrans = (Atrans ? 'T' : 'N');
            Rtrans = (Btrans ? 'T' : 'N');
            Llda = (Atrans ? isize : ksize); 
            Rlda = (Btrans ? ksize : jsize); 
        }
        size_t nzip = ksize;
        size_t Clda = (Ctrans ? isize : jsize);

        /// NICK: Need the following math_zgemv, math_zdot, math_zger

        if (nrow == 1L && ncol == 1L && nzip == 1L) {
            (*C2p) = alpha * (*Lp) * (*Rp) + beta * (*C2p);
        } else if (nrow == 1L && ncol == 1L && nzip > 1L) {
            (*C2p) *= beta;
            // (*C2p) += alpha * C_DDOT(nzip, Lp, 1, Rp, 1);
            (*C2p) += alpha * math_zdot(nzip, Lp, 1, Rp, 1);
        } else if (nrow == 1L && ncol > 1L && nzip == 1L) {
            // C_DSCAL(ncol, beta, C2p, 1);
            math_zscale(
                ncol, 
                beta, 
                C2p, 
                1);
            // C_DAXPY(ncol, alpha * (*Lp), Rp, 1, C2p, 1);
            math_zaxpy(
                ncol,
                alpha * (*Lp), 
                Rp, 
                1, 
                C2p, 
                1);
        } else if (nrow > 1L && ncol == 1L && nzip == 1L) {
            // C_DSCAL(nrow, beta, C2p, 1);
            math_zscale(
                nrow, 
                beta, 
                C2p, 
                1);
            // C_DAXPY(nrow, alpha * (*Rp), Lp, 1, C2p, 1);
            math_zaxpy(
                nrow, 
                alpha * (*Rp), 
                Lp, 
                1, 
                C2p, 
                1);
        } else if (nrow > 1L && ncol > 1L && nzip == 1L) {
            for (size_t row = 0L; row < nrow; row++) {
                // C_DSCAL(ncol, beta, C2p + row * Clda, 1);
                math_zscale(
                    ncol, 
                    beta, 
                    C2p + row * Clda, 
                    1);
            }
            // C_DGER(nrow, ncol, alpha, Lp, 1, Rp, 1, C2p, Clda);
            math_zger(
                nrow, 
                ncol, 
                alpha, 
                Lp, 
                1, 
                Rp, 
                1, 
                C2p, 
                Clda);
        } else if (nrow == 1 && ncol > 1 && nzip > 1) {
            if (Rtrans == 'N') {
                // C_DGEMV('T', nzip, ncol, alpha, Rp, Rlda, Lp, 1, beta, C2p, 1);
                math_zgemv(
                    'T', 
                    nzip, 
                    ncol, 
                    alpha, 
                    Rp, 
                    Rlda, 
                    Lp, 
                    1, 
                    beta, 
                    C2p, 
                    1);
            } else {
                // C_DGEMV('N', ncol, nzip, alpha, Rp, Rlda, Lp, 1, beta, C2p, 1);
                math_zgemv(
                    'N', 
                    ncol, 
                    nzip, 
                    alpha, 
                    Rp, 
                    Rlda, 
                    Lp, 
                    1, 
                    beta, 
                    C2p, 
                    1);
            }
        } else if (nrow > 1 && ncol == 1 && nzip > 1) {
            if (Ltrans == 'N') {
                // C_DGEMV('N', nrow, nzip, alpha, Lp, Llda, Rp, 1, beta, C2p, 1);
                std::cout << "I get to math_zgemv with nrow > 1 && ncol == 1 && nzip > 1 and Ltrans == 'N'" << std::endl;
                math_zgemv(
                    'N', 
                    nrow, 
                    nzip, 
                    alpha, 
                    Lp, 
                    Llda, 
                    Rp, 
                    1, 
                    beta, 
                    C2p, 
                    1);
            } else {
                // C_DGEMV('T', nzip, nrow, alpha, Lp, Llda, Rp, 1, beta, C2p, 1);
                std::cout << "I get to math_zgemv with nrow > 1 && ncol == 1 && nzip > 1 and Ltrans == 'T'" << std::endl;
                math_zgemv(
                    'T', 
                    nzip, 
                    nrow, 
                    alpha, 
                    Lp, 
                    Llda, 
                    Rp, 
                    1, 
                    beta, 
                    C2p, 
                    1);
            }
        } else {
            // C_DGEMM(Ltrans, Rtrans, nrow, ncol, nzip, alpha, Lp, Llda, Rp, Rlda, beta, C2p, Clda);
            std::cout << "I get to zgemm!" << std::endl;

            math_zgemm(
                Ltrans, 
                Rtrans, 
                nrow, 
                ncol, 
                nzip, 
                alpha, 
                Lp, 
                Llda, 
                Rp, 
                Rlda, 
                beta, 
                C2p, 
                Clda);
        }

        C2p += isize * jsize;
        A2p += ksize * isize;
        B2p += ksize * jsize;
    }

    // => Result Permutation <= //

    if (Cperm) {
        // Tensor::permute(Cinds2,Cinds,C2,C);
        // C = Tensor::permute(Cinds2, Cinds, C2, C);
        // Tensor::permute(Cinds2, Cinds, C2, C);
        Tensor::permute(Cinds2, Cinds, C2, C3);
    } else {
        C3 = C2;
    }

    // std::cout << "\n end: C  \n" << C.str() << std::endl;
    // std::cout << "\n end: C2 \n" << C2.str() << std::endl;
    // std::cout << "\n end: C3 \n" << C3.str() << std::endl;

    // return C;
    // C3 = C;
    // return;
}

// } // namespace lightspeed