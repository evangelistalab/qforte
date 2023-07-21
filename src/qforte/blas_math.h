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
}

#endif // _blas_math_h_