#ifndef _blas_math_h_
#define _blas_math_h_

extern "C" {
  // Function declaration for BLAS DAXPY function
  // daxpy: y = alpha * x + y
  void daxpy(const int n, const double alpha, const double* x, const int incx, double* y, const int incy);
}

#endif // _blas_math_h_