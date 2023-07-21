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
    cblas_zaxpy(n, &alpha, reinterpret_cast<const double*>(x), incx, reinterpret_cast<double*>(y), incy);
}