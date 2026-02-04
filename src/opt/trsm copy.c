/*
 * trsm.c
 * High-Performance Implementation of Blocked Dense Linear Algebra Kernels
 * Generated based on gemini.md
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

// -----------------------------------------------------------------------------
// Constants and Macros
// -----------------------------------------------------------------------------

#define MR 6
#define NR 8
#define MC 256
#define KC 256
#define NC 1024
#define TRSM_BLK 128

// -----------------------------------------------------------------------------
// Micro-Kernel
// -----------------------------------------------------------------------------

/*
 * micro_kernel_dgemm
 * Computes C_sub += alpha * A_packed * B_packed
 *
 * A_packed: contiguous buffer of size k * MR
 * B_packed: contiguous buffer of size k * NR
 * C_sub: sub-matrix of C with stride ldc
 */
void micro_kernel_dgemm(int k, double alpha, const double *restrict A,
                        const double *restrict B, double *restrict C, int ldc) {
  // Local accumulators for C (stored in registers)
  double ab[MR][NR];

  // Clear accumulators
  for (int i = 0; i < MR; ++i) {
    for (int j = 0; j < NR; ++j) {
      ab[i][j] = 0.0;
    }
  }

  // Main Compute Loop (Rank-1 updates)
  for (int l = 0; l < k; ++l) {
    for (int j = 0; j < NR; ++j) {
      // Load B element once and broadcast logically
      // B is packed as k x NR (row-major inside panel)
      double b_val = B[l * NR + j];

      for (int i = 0; i < MR; ++i) {
        // Fused Multiply-Add
        // A is packed in column-major order within the micropanel (MR x k)
        // Index for A(i, l) is l*MR + i
        ab[i][j] += A[l * MR + i] * b_val;
      }
    }
  }

  // Write back to C (Alpha scaling and Beta accumulation)
  for (int j = 0; j < NR; ++j) {
    for (int i = 0; i < MR; ++i) {
      C[i + j * ldc] += alpha * ab[i][j];
    }
  }
}

// -----------------------------------------------------------------------------
// Packing Routines
// -----------------------------------------------------------------------------

/*
 * pack_A
 * Packs a panel of A into a buffer.
 */
void pack_A(int mc, int kc, const double *A, int incRow, int incCol,
            double *buffer) {
  int mp = mc / MR;     // Number of full micropanels
  int mr_rem = mc % MR; // Remainder

  for (int i = 0; i < mp; ++i) {
    // Pack one micropanel of size MR x kc
    for (int k = 0; k < kc; ++k) {
      for (int r = 0; r < MR; ++r) {
        *buffer++ = A[r * incRow + k * incCol];
      }
    }
    A += MR * incRow; // Advance A to next block of rows
  }

  // Handle remainder (edge case)
  if (mr_rem > 0) {
    for (int k = 0; k < kc; ++k) {
      for (int r = 0; r < mr_rem; ++r) {
        *buffer++ = A[r * incRow + k * incCol];
      }
      // Pad the rest of the micropanel with zeros
      for (int r = mr_rem; r < MR; ++r) {
        *buffer++ = 0.0;
      }
    }
  }
}

/*
 * pack_B
 * Packs a panel of B into a buffer.
 */
void pack_B(int kc, int nc, const double *B, int incRow, int incCol,
            double *buffer) {
  int np = nc / NR;
  int nr_rem = nc % NR;

  for (int j = 0; j < np; ++j) {
    // Pack micropanel of size kc x NR
    for (int k = 0; k < kc; ++k) {
      for (int c = 0; c < NR; ++c) {
        *buffer++ = B[k * incRow + c * incCol];
      }
    }
    B += NR * incCol; // Advance B to next block of cols
  }

  if (nr_rem > 0) {
    for (int k = 0; k < kc; ++k) {
      for (int c = 0; c < nr_rem; ++c) {
        *buffer++ = B[k * incRow + c * incCol];
      }
      for (int c = nr_rem; c < NR; ++c) {
        *buffer++ = 0.0;
      }
    }
  }
}

// -----------------------------------------------------------------------------
// DGEMM Drivers
// -----------------------------------------------------------------------------

void dgemm_blocked_driver(int m, int n, int k, double alpha, const double *A,
                          int incRowA, int incColA, const double *B,
                          int incRowB, int incColB, double beta, double *C,
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
    double *bufA = aligned_alloc(64, MC * KC * sizeof(double));
    double *bufB = aligned_alloc(64, KC * NC * sizeof(double));

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
            for (int ir = 0; ir < mc_cur; ir += MR) {
              micro_kernel_dgemm(
                  kc_cur, alpha,
                  bufA + ir * kc_cur, // Pointer arithmetic for packed A
                  bufB + jr * kc_cur, // Pointer arithmetic for packed B
                  C + (ic + ir) + (jc + jr) * ldc, ldc);
            }
          }
        }
      }
    }

    free(bufA);
    free(bufB);
  }
}

void dgemm_scalar_direct(int m, int n, int k, double alpha, const double *A,
                         int rsa, int csa, const double *B, int rsb, int csb,
                         double beta, double *C, int ldc) {
  // Loops ordered J, I, L for generic C efficiency
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      double sum = 0.0;
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

void dgemm_n1(int m, int k, double alpha, const double *A, int rsa, int csa,
              const double *B, int rsb, // csb irrelevant as N=1
              double beta, double *C) {
  for (int i = 0; i < m; ++i) {
    double sum = 0.0;
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

void my_dgemm(char transa, char transb, int m, int n, int k, double alpha,
              const double *A, int lda, const double *B, int ldb, double beta,
              double *C, int ldc) {
  // 1. Stride Setup
  int rsa = (transa == 'N' || transa == 'n') ? 1 : lda;
  int csa = (transa == 'N' || transa == 'n') ? lda : 1;
  int rsb = (transb == 'N' || transb == 'n') ? 1 : ldb;
  int csb = (transb == 'N' || transb == 'n') ? ldb : 1;

  // 2. Specialization Dispatch
  if (m == 1 && n == 1) {
    // Dot product
    double sum = 0.0;
    for (int l = 0; l < k; ++l)
      sum += A[l * csa] * B[l * rsb];
    if (beta == 0.0)
      *C = alpha * sum;
    else
      *C = alpha * sum + beta * *C;
    return;
  }

  if (n == 1) {
    dgemm_n1(m, k, alpha, A, rsa, csa, B, rsb, beta, C);
    return;
  }

  // 3. Threshold Check
  long long ops = (long long)m * n * k;
  if (ops < 20000) { // Tunable threshold
    dgemm_scalar_direct(m, n, k, alpha, A, rsa, csa, B, rsb, csb, beta, C, ldc);
    return;
  }

  // 4. Heavyweight Driver
  dgemm_blocked_driver(m, n, k, alpha, A, rsa, csa, B, rsb, csb, beta, C, ldc);
}

// -----------------------------------------------------------------------------
// DTRSM Implementation
// -----------------------------------------------------------------------------

// Helper: Small serial TRSM for diagonal blocks
void dtrsm_small(char side, char uplo, char transa, char diag, int m, int n,
                 double alpha, const double *A, int lda, double *B, int ldb) {
  // Scaling B by alpha first
  if (alpha != 1.0) {
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i)
        B[i + j * ldb] *= alpha;
  }

  int lside = (side == 'L' || side == 'l');
  int lower = (uplo == 'L' || uplo == 'l');
  int unit = (diag == 'U' || diag == 'u');
  int trans =
      (transa == 'T' || transa == 't' || transa == 'C' || transa == 'c');

  if (lside) {
    // Left Side: op(A) * X = B
    for (int j = 0; j < n; ++j) {
      if (!trans) {
        if (lower) {
          // Forward substitution
          for (int i = 0; i < m; ++i) {
            for (int k = 0; k < i; ++k) {
              B[i + j * ldb] -= A[i + k * lda] * B[k + j * ldb];
            }
            if (!unit)
              B[i + j * ldb] /= A[i + i * lda];
          }
        } else { // Upper
          // Backward substitution
          for (int i = m - 1; i >= 0; --i) {
            for (int k = i + 1; k < m; ++k) {
              B[i + j * ldb] -= A[i + k * lda] * B[k + j * ldb];
            }
            if (!unit)
              B[i + j * ldb] /= A[i + i * lda];
          }
        }
      } else {       // Transposed
        if (lower) { // Lower Transposed -> effectively Upper
          // Backward
          for (int i = m - 1; i >= 0; --i) {
            for (int k = i + 1; k < m; ++k) {
              B[i + j * ldb] -= A[k + i * lda] * B[k + j * ldb];
            }
            if (!unit)
              B[i + j * ldb] /= A[i + i * lda];
          }
        } else { // Upper Transposed -> effectively Lower
          // Forward
          for (int i = 0; i < m; ++i) {
            for (int k = 0; k < i; ++k) {
              B[i + j * ldb] -= A[k + i * lda] * B[k + j * ldb];
            }
            if (!unit)
              B[i + j * ldb] /= A[i + i * lda];
          }
        }
      }
    }
  } else {
    // Right Side: X * op(A) = B
    for (int i = 0; i < m; ++i) {
      if (!trans) {
        if (lower) {
          // Backward substitution (L^T is Upper)
          for (int k = n - 1; k >= 0; --k) {
            for (int j = k + 1; j < n; ++j) {
              B[i + k * ldb] -= B[i + j * ldb] * A[j + k * lda];
            }
            if (!unit)
              B[i + k * ldb] /= A[k + k * lda];
          }
        } else { // Upper
          // Forward substitution (U^T is Lower)
          for (int k = 0; k < n; ++k) {
            for (int j = 0; j < k; ++j) {
              B[i + k * ldb] -= B[i + j * ldb] * A[j + k * lda];
            }
            if (!unit)
              B[i + k * ldb] /= A[k + k * lda];
          }
        }
      } else { // Transposed
        if (lower) {
          // Forward
          for (int k = 0; k < n; ++k) {
            for (int j = 0; j < k; ++j) {
              B[i + k * ldb] -= B[i + j * ldb] * A[k + j * lda];
            }
            if (!unit)
              B[i + k * ldb] /= A[k + k * lda];
          }
        } else {
          // Backward
          for (int k = n - 1; k >= 0; --k) {
            for (int j = k + 1; j < n; ++j) {
              B[i + k * ldb] -= B[i + j * ldb] * A[k + j * lda];
            }
            if (!unit)
              B[i + k * ldb] /= A[k + k * lda];
          }
        }
      }
    }
  }
}

void my_dtrsm(char side, char uplo, char transa, char diag, int m, int n,
              double alpha, const double *A, int lda, double *B, int ldb) {
  // Fast Path for Small Matrices
  if (m <= TRSM_BLK && n <= TRSM_BLK) {
    dtrsm_small(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
    return;
  }

  int lside = (side == 'L' || side == 'l');
  int lower = (uplo == 'L' || uplo == 'l');
  int trans =
      (transa == 'T' || transa == 't' || transa == 'C' || transa == 'c');

  // Determine Logic:
  int forward = 1;
  if (lside) {
    // Logical Lower is Lower && !Trans, OR Upper && Trans
    int logical_lower = (lower && !trans) || (!lower && trans);
    forward = logical_lower;
  } else {
    // Right Side Logic
    int logical_lower = (lower && !trans) || (!lower && trans);
    forward = !logical_lower;
  }

#pragma omp parallel
  {
#pragma omp single
    {
      // Main Block Loop
      int limit = lside ? m : n;

      for (int k_idx = 0; k_idx < limit; k_idx += TRSM_BLK) {
        // Calculate actual k based on direction
        int k = forward ? k_idx : (limit - k_idx - TRSM_BLK);
        if (k < 0)
          k = 0; // Boundary fix

        int blk_size = TRSM_BLK;
        if (k + blk_size > limit)
          blk_size = limit - k;

        // 1. Task: Solve Diagonal Block
        double *dep_token = lside ? &B[k] : &B[k * ldb];

#pragma omp task depend(inout : dep_token) firstprivate(k, blk_size)
        {
          const double *Ak;
          double *Bk;
          if (lside) {
            Ak = &A[k + k * lda];
            Bk = &B[k]; // B row k
            dtrsm_small(side, uplo, transa, diag, blk_size, n, alpha, Ak, lda,
                        Bk, ldb);
          } else {
            Ak = &A[k + k * lda];
            Bk = &B[k * ldb]; // B col k
            dtrsm_small(side, uplo, transa, diag, m, blk_size, alpha, Ak, lda,
                        Bk, ldb);
          }
        }

        // 2. Task: Update Off-Diagonal
        if (forward) {
          for (int i = k + blk_size; i < limit; i += TRSM_BLK) {
            int i_size = (i + TRSM_BLK > limit) ? limit - i : TRSM_BLK;

            double *dep_src = lside ? &B[k] : &B[k * ldb];
            double *dep_dst = lside ? &B[i] : &B[i * ldb];

#pragma omp task depend(in : dep_src) depend(inout : dep_dst)                  \
    firstprivate(i, k, blk_size, i_size)
            {
              // GEMM Update
              if (lside) {
                // B[i] -= A[i,k] * B[k]
                my_dgemm('N', 'N', i_size, n, blk_size, -1.0, &A[i + k * lda],
                         lda, &B[k], ldb, 1.0, &B[i], ldb);
              } else {
                // B[i] -= B[k] * A[k,i]
                my_dgemm('N', 'N', m, i_size, blk_size, -1.0, &B[k * ldb], ldb,
                         &A[k + i * lda], lda, 1.0, &B[i * ldb], ldb);
              }
            }
          }
        } else {
          // Backward Loop
          for (int temp_i = k - TRSM_BLK;; temp_i -= TRSM_BLK) {
            int i = temp_i;
            int i_size = TRSM_BLK;
            if (i < 0) {
              i = 0;
              i_size = k; // The gap from 0 to k
            }

            if (i_size > 0) {
              double *dep_src = lside ? &B[k] : &B[k * ldb];
              double *dep_dst = lside ? &B[i] : &B[i * ldb];

#pragma omp task depend(in : dep_src) depend(inout : dep_dst)                  \
    firstprivate(i, k, blk_size, i_size)
              {
                if (lside) {
                  my_dgemm('N', 'N', i_size, n, blk_size, -1.0, &A[i + k * lda],
                           lda, &B[k], ldb, 1.0, &B[i], ldb);
                } else {
                  my_dgemm('N', 'N', m, i_size, blk_size, -1.0, &B[k * ldb],
                           ldb, &A[k + i * lda], lda, 1.0, &B[i * ldb], ldb);
                }
              }
            }

            if (i == 0)
              break;
          }
        }
      }
    }
  }
}
