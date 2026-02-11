#include <omp.h>

typedef double real;
#define BLASNAME(f) d##f##_

#define MR 8
#define NR 8

#define MC 256
#define KC 256
#define NC 1024 // Adjusted to L3 size per thread

#include "gemm_impl.i"

#define TRSM_BLK 128

#include "trsm_impl.i"
