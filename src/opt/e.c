/* 
 * Micro-Kernel parameters
 * Targeted for AVX2 (Double Precision)
 */
#define MR 4
#define NR 4

#define MC 128
#define KC 128
#define NC 512 // Adjusted to L3 size per thread

typedef long double real;
#define BLASNAME(f) e##f##_

#include "gemm_impl.i"


#define TRSM_BLK 64

#include "trsm_impl.i"