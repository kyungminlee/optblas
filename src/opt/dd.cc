#include <omp.h>
#include "Float64x2.hh"

typedef float64x2 real;
#define BLASNAME(f) dd##f##_

#define MR 4
#define NR 4

#define MC 128
#define KC 128
#define NC 128 // Adjusted to L3 size per thread

#define RESTRICT __restrict

#include "gemm_impl.i"

// #define TRSM_BLK 128

// #include "trsm_impl.i"
