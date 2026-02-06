/* 
 * Micro-Kernel parameters
 * Targeted for AVX2 (Double Precision)
 */
#define MR 8
#define NR 8

#define MC 256
#define KC 256
#define NC 1024 // Adjusted to L3 size per thread

typedef double real;
#define BLASNAME(f) d##f##_

#include "gemm_impl.i"