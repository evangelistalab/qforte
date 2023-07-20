#include "blas_math.h"

extern "C" {
  // Include the actual BLAS library header
  #include <cblas.h>
}

void daxpy(const int n, const double alpha, const double* x, const int incx, double* y, const int incy) {
  // Call the BLAS DAXPY function
  cblas_daxpy(n, alpha, x, incx, y, incy);
}