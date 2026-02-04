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


#define TRSM_BLK 128

void my_dgemm(char transa, char transb, int m, int n, int k, double alpha,
              const double *A, int lda, const double *B, int ldb, double beta,
              double *C, int ldc);


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
              B[i + k * ldb] -= B[i + j * ldb] * A[k + j * lda];
            }
            if (!unit)
              B[i + k * ldb] /= A[k + k * lda];
          }
        } else { // Upper
          // Forward substitution (U^T is Lower)
          for (int k = 0; k < n; ++k) {
            for (int j = 0; j < k; ++j) {
              B[i + k * ldb] -= B[i + j * ldb] * A[k + j * lda];
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
              B[i + k * ldb] -= B[i + j * ldb] * A[j + k * lda];
            }
            if (!unit)
              B[i + k * ldb] /= A[k + k * lda];
          }
        } else {
          // Backward
          for (int k = n - 1; k >= 0; --k) {
            for (int j = k + 1; j < n; ++j) {
              B[i + k * ldb] -= B[i + j * ldb] * A[j + k * lda];
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
        int k, blk_size;
        if (forward) {
          k = k_idx;
          blk_size = (k + TRSM_BLK > limit) ? limit - k : TRSM_BLK;
        } else {
          int block_idx = (k_idx / TRSM_BLK);
          int num_blocks = (limit + TRSM_BLK - 1) / TRSM_BLK;
          int current_block = num_blocks - 1 - block_idx;
          k = current_block * TRSM_BLK;
          blk_size = (k + TRSM_BLK > limit) ? limit - k : TRSM_BLK;
        }

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


void dtrsm_(char const * side, char const * uplo, char const * transa, char const * diag, int const * m, int const * n,
            double const * alpha, const double * restrict A, int const * lda, double * restrict B, int const * ldb) {
  my_dtrsm(*side, *uplo, *transa, *diag, *m, *n, *alpha, A, *lda, B, *ldb);
}