#ifndef _blas_math_h_
#define _blas_math_h_

#include "qforte-def.h"

extern "C" {
  // Function declaration for BLAS DAXPY function
  // daxpy: y = alpha * x + y
  void math_daxpy(
    const int n, 
    const double alpha, 
    const double* x, 
    const int incx, 
    double* y, 
    const int incy);

  // Function declaration for BLAS ZAXPY function (complex version of DAZPY)
  // daxpy: y = alpha * x + y
  void math_zaxpy(
    const int n,
    const std::complex<double> alpha,
    const std::complex<double>* x,
    const int incx,
    std::complex<double>* y,
    const int incy);

  // Function declaration for BLAS ZSCALE function (complex version of DSCAL)
  // zscale: x = alpha * x
  void math_zscale(
    const int n,
    const std::complex<double> alpha,
    std::complex<double>* x,
    const int incx);

  // Function declaration for BLAS ZGEMM function (complex version of DGEMM)
  // zgemm: C = alpha * op(A) * op(B) + beta * C
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
    const int ldc);
}

#endif // _blas_math_h_