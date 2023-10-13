#include "blas_math.h"
#include <complex>

extern "C" {
  // Include the actual BLAS library header
  #include <cblas.h>
}

void math_daxpy(
  const int n, 
  const double alpha, 
  const double* x, 
  const int incx, 
  double* y, 
  const int incy) 
{
  // Call the BLAS DAXPY function
  cblas_daxpy(n, alpha, x, incx, y, incy);
}

void math_zaxpy(
    const int n,
    const std::complex<double> alpha,
    const std::complex<double>* x,
    const int incx,
    std::complex<double>* y,
    const int incy) 
{   
    
    // Call the BLAS ZAXPY function
    cblas_zaxpy(
      n, 
      &alpha, 
      reinterpret_cast<const openblas_complex_double*>(x), 
      incx, 
      reinterpret_cast<openblas_complex_double*>(y), 
      incy);
}

void math_zscale(
    const int n,
    const std::complex<double> alpha,
    std::complex<double>* x,
    const int incx)
{
    // Call the CBLAS ZSCALE function
    cblas_zscal(
      n, 
      &alpha, 
      reinterpret_cast<openblas_complex_double*>(x), 
      incx);
}

void math_zgemm(
    const char transa,
    const char transb,
    const int M,
    const int N,
    const int K,
    const std::complex<double> alpha,
    const std::complex<double>* A,
    const int lda,
    const std::complex<double>* B,
    const int ldb,
    const std::complex<double> beta,
    std::complex<double>* C,
    const int ldc)
{
    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;

    if(transa == 'T'){ transA = CblasTrans; }
    if(transa == 'C'){ transA = CblasConjTrans; }

    if(transb == 'T'){ transB = CblasTrans; }
    if(transb == 'C'){ transB = CblasConjTrans; }

    cblas_zgemm(
      CblasRowMajor, // was CblasColMajor Should be CblasRowMajor
      transA, 
      transB, 
      M, 
      N, 
      K, 
      reinterpret_cast<const openblas_complex_double*>(&alpha), 
      reinterpret_cast<const openblas_complex_double*>(A), 
      lda, 
      reinterpret_cast<const openblas_complex_double*>(B), 
      ldb, 
      reinterpret_cast<const openblas_complex_double*>(&beta), 
      reinterpret_cast<openblas_complex_double*>(C), 
      ldc);
}

void math_zgemv(
    const char trans,
    const int M,
    const int N,
    const std::complex<double> alpha,
    const std::complex<double>* A,
    const int lda,
    const std::complex<double>* x,
    const int incx,
    const std::complex<double> beta,
    std::complex<double>* y,
    const int incy)
{
    CBLAS_TRANSPOSE transA = CblasNoTrans;

    if (trans == 'T') { 
        transA = CblasTrans; 
    } else if (trans == 'C') { 
        transA = CblasConjTrans; 
    }

    cblas_zgemv(
        CblasRowMajor, 
        transA, 
        M, 
        N, 
        reinterpret_cast<const openblas_complex_double*>(&alpha), 
        reinterpret_cast<const openblas_complex_double*>(A), 
        lda, 
        reinterpret_cast<const openblas_complex_double*>(x), 
        incx, 
        reinterpret_cast<const openblas_complex_double*>(&beta), 
        reinterpret_cast<openblas_complex_double*>(y), 
        incy);
}

std::complex<double> math_zdot(
    const int n,
    const std::complex<double>* x,
    const int incx,
    const std::complex<double>* y,
    const int incy)
{

  std::complex<double> result;

    /// NICK: Odd that this doesn't have an conjugate option
    cblas_zdotu_sub(
        n, 
        reinterpret_cast<const openblas_complex_double*>(x), 
        incx, 
        reinterpret_cast<const openblas_complex_double*>(y), 
        incy,
        reinterpret_cast<openblas_complex_double*>(&result));

    return result;
}

void math_zger(
    const int m,
    const int n,
    const std::complex<double> alpha,
    const std::complex<double>* x,
    const int incx,
    const std::complex<double>* y,
    const int incy,
    std::complex<double>* A,
    const int lda)
{
    cblas_zgeru(
        CblasRowMajor, 
        m, 
        n, 
        reinterpret_cast<const openblas_complex_double*>(&alpha), 
        reinterpret_cast<const openblas_complex_double*>(x), 
        incx, 
        reinterpret_cast<const openblas_complex_double*>(y), 
        incy, 
        reinterpret_cast<openblas_complex_double*>(A), 
        lda);
}
