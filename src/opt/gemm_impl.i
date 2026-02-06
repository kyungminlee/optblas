#pragma once
#include <stdlib.h>

#ifndef BLASNAME
#include "dgemm.c"
#endif

/*
 * micro_kernel_PREFIX##gemm
 * Computes C_sub += alpha * A_packed * B_packed
 * 
 * A_packed: contiguous buffer of size k * MR
 * B_packed: contiguous buffer of size k * NR
 * C_sub: sub-matrix of C with stride ldc
 */
static void micro_kernel_gemm(
    int k, 
    real alpha, 
    const real * restrict A, 
    const real * restrict B, 
    real * restrict C, 
    int ldc,
    int mr_cur,
    int nr_cur)
{
    // Local accumulators for C (stored in registers)
    real ab[MR][NR];

    // Clear accumulators
    for(int i = 0; i < MR; ++i) {
        for(int j = 0; j < NR; ++j) {
            ab[i][j] = 0.0;
        }
    }

    // Main Compute Loop (Rank-1 updates)
    for(int l = 0; l < k; ++l) {
        // Prefetching hints could be placed here
        __builtin_prefetch(A + MR * (l+1));
        __builtin_prefetch(B + NR * (l+1));

        for(int j = 0; j < nr_cur; ++j) {
            // Load B element once and broadcast logically
            real b_val = B[l * NR + j]; 

            for(int i = 0; i < mr_cur; ++i) {
                // Fused Multiply-Add
                // A is packed in column-major order within the micropanel
                ab[i][j] += A[l * MR + i] * b_val;
            }
        }
    }

    // Write back to C (Alpha scaling and Beta accumulation)
    // Note: Beta is usually handled in the macro-kernel loop before this call
    // or we assume beta=1 here and handle beta=0 separately. 
    // The standard micro-kernel updates: C = C + alpha * AB
    for(int j = 0; j < nr_cur; ++j) {
        for(int i = 0; i < mr_cur; ++i) {
            C[i + j * ldc] += alpha * ab[i][j];
        }
    }
}


/*
 * pack_A
 * Packs a panel of A into a buffer.
 * The buffer is organized as a sequence of micropanels (MR x kc).
 *
 * mc: number of rows to pack
 * kc: number of columns to pack
 * A: source pointer
 * incRow, incCol: strides of A
 * buffer: destination
 */
static void pack_A(int mc, int kc, const real *A, int incRow, int incCol,
            real *buffer) {
  int mp = mc / MR;     // Number of full micropanels
  int mr_rem = mc % MR; // Remainder

  for (int i = 0; i < mp; ++i) {
    // Pack one micropanel of size MR x kc
    for (int k = 0; k < kc; ++k) {
      for (int r = 0; r < MR; ++r) {
        // Depending on the micro-kernel expectation,
        // we typically store A elements contiguously down the column of the
        // micropanel
        *buffer++ = A[(i * MR + r) * incRow + k * incCol];
      }
    }
  }

  // Handle remainder (edge case)
  if (mr_rem > 0) {
    for (int k = 0; k < kc; ++k) {
      for (int r = 0; r < mr_rem; ++r) {
        *buffer++ = A[(mp * MR + r) * incRow + k * incCol];
      }
      // Pad the rest of the micropanel with zeros to avoid segfaults in
      // micro-kernel
      for (int r = mr_rem; r < MR; ++r) {
        *buffer++ = 0.0;
      }
    }
  }
}

static
void pack_B(int kc, int nc, const real *B, int incRow, int incCol,
            real *buffer) {
  int np = nc / NR;
  int nr_rem = nc % NR;

  for (int j = 0; j < np; ++j) {
    // Pack micropanel of size kc x NR
    for (int k = 0; k < kc; ++k) {
      for (int c = 0; c < NR; ++c) {
        // B is typically stored row-wise in the packed buffer for efficient
        // broadcast
        *buffer++ = B[k * incRow + (j * NR + c) * incCol];
      }
    }
  }

  if (nr_rem > 0) {
    for (int k = 0; k < kc; ++k) {
      for (int c = 0; c < nr_rem; ++c) {
        *buffer++ = B[k * incRow + (np * NR + c) * incCol];
      }
      for (int c = nr_rem; c < NR; ++c) {
        *buffer++ = 0.0;
      }
    }
  }
}


static
void gemm_blocked_driver(int m, int n, int k, real alpha, const real *A,
                          int incRowA, int incColA, const real *B,
                          int incRowB, int incColB, real beta, real *C,
                          int ldc) {
  // Handle Beta Scaling first (Parallel)
  if (beta != 1.0) {
#pragma omp parallel for collapse(2)
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < m; ++i) {
        if (beta == 0.0)
          C[i + j * ldc] = 0.0;
        else
          C[i + j * ldc] *= beta;
      }
    }
  }

  if (alpha == 0.0)
    return;

// Parallelize Loop 5 (jc)
#pragma omp parallel
  {
    // Thread-private packing buffers
    real *bufA = aligned_alloc(64, MC * KC * sizeof(real));
    real *bufB = aligned_alloc(64, KC * NC * sizeof(real));

#pragma omp for schedule(dynamic)
    for (int jc = 0; jc < n; jc += NC) {
      int nc_cur = (n - jc < NC) ? (n - jc) : NC;

      // Loop 4 (pc)
      for (int pc = 0; pc < k; pc += KC) {
        int kc_cur = (k - pc < KC) ? (k - pc) : KC;

        // Pack B panel
        pack_B(kc_cur, nc_cur, B + pc * incRowB + jc * incColB, incRowB,
               incColB, bufB);

        // Loop 3 (ic)
        for (int ic = 0; ic < m; ic += MC) {
          int mc_cur = (m - ic < MC) ? (m - ic) : MC;

          // Pack A panel
          pack_A(mc_cur, kc_cur, A + ic * incRowA + pc * incColA, incRowA,
                 incColA, bufA);

          // Loop 2 & 1 (Micro-kernel iteration)
          for (int jr = 0; jr < nc_cur; jr += NR) {
            int nr = (nc_cur - jr < NR) ? (nc_cur - jr) : NR;
            for (int ir = 0; ir < mc_cur; ir += MR) {
              int mr = (mc_cur - ir < MR) ? (mc_cur - ir) : MR;
              micro_kernel_gemm(
                  kc_cur, alpha,
                  bufA + ir * kc_cur, // Pointer arithmetic for packed A
                  bufB + jr * kc_cur, // Pointer arithmetic for packed B
                  C + (ic + ir) + (jc + jr) * ldc, ldc, mr, nr);
            }
          }
        }
      }
    }

    free(bufA);
    free(bufB);
  }
}

static
void gemm_scalar_direct(int m, int n, int k, real alpha, const real *A,
                         int rsa, int csa, const real *B, int rsb, int csb,
                         real beta, real *C, int ldc) {
  // Loops ordered J, I, L for generic C efficiency
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      real sum = 0.0;
      for (int l = 0; l < k; ++l) {
        sum += A[i * rsa + l * csa] * B[l * rsb + j * csb];
      }
      if (beta == 0.0)
        C[i + j * ldc] = alpha * sum;
      else
        C[i + j * ldc] = alpha * sum + beta * C[i + j * ldc];
    }
  }
}

static
void gemm_n1(int m, int k, real alpha, const real *A, int rsa, int csa,
              const real *B, int rsb, // csb irrelevant as N=1
              real beta, real *C) {
  for (int i = 0; i < m; ++i) {
    real sum = 0.0;
// Vectorize this loop
#pragma omp simd reduction(+ : sum)
    for (int l = 0; l < k; ++l) {
      sum += A[i * rsa + l * csa] * B[l * rsb];
    }
    if (beta == 0.0)
      C[i] = alpha * sum;
    else
      C[i] = alpha * sum + beta * C[i];
  }
}


void BLASNAME(gemm_opt)(char transa, char transb, int m, int n, int k, real alpha,
              const real *A, int lda, const real *B, int ldb, real beta,
              real *C, int ldc) {
  // 1. Stride Setup
  int rsa = (transa == 'N' || transa == 'n') ? 1 : lda;
  int csa = (transa == 'N' || transa == 'n') ? lda : 1;
  int rsb = (transb == 'N' || transb == 'n') ? 1 : ldb;
  int csb = (transb == 'N' || transb == 'n') ? ldb : 1;

  // 2. Specialization Dispatch
  if (m == 1 && n == 1) {
    // Dot product
    real sum = 0.0;
    for (int l = 0; l < k; ++l)
      sum += A[l * csa] * B[l * rsb];
    if (beta == 0.0)
      *C = alpha * sum;
    else
      *C = alpha * sum + beta * *C;
    return;
  }

  if (n == 1) {
    gemm_n1(m, k, alpha, A, rsa, csa, B, rsb, beta, C);
    return;
  }

  // 3. Threshold Check
  long long ops = (long long)m * n * k;
  if (ops < 20000) { // Tunable threshold
    gemm_scalar_direct(m, n, k, alpha, A, rsa, csa, B, rsb, csb, beta, C, ldc);
    return;
  }

  // 4. Heavyweight Driver
  gemm_blocked_driver(m, n, k, alpha, A, rsa, csa, B, rsb, csb, beta, C, ldc);
}

void BLASNAME(gemm)(char const * ptransa, char const * ptransb, int const * pm, int const * pn, int const * pk, real const * palpha,
              const real * restrict A, int const * plda, const real * restrict B, int const * pldb, real const * pbeta,
              real * restrict C, int const * pldc) {
  BLASNAME(gemm_opt)(*ptransa, *ptransb, *pm, *pn, *pk, *palpha, A, *plda, B, *pldb, *pbeta, C, *pldc);
}
