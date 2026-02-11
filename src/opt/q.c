#include <omp.h>
#include <quadmath.h>

typedef __float128 real;
#define BLASNAME(f) q##f##_

#define MR 4
#define NR 4

#define MC 128
#define KC 128
#define NC 512 // Adjusted to L3 size per thread

#include "gemm_impl.i"

#define TRSM_BLK 64

#include "trsm_impl.i"