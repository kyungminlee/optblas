/*
 * trsm.c
 * High-Performance Implementation of Blocked Dense Linear Algebra Kernels
 * Generated based on gemini.md
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <string.h>


// -----------------------------------------------------------------------------
// Constants and Macros
// -----------------------------------------------------------------------------

// static
// void gemm_opt(char transa, char transb, int m, int n, int k, real alpha,
//               const real *A, int lda, const real *B, int ldb, real beta,
//               real *C, int ldc);

// -----------------------------------------------------------------------------
// DTRSM Implementation
// -----------------------------------------------------------------------------

// Helper: Reference DTRSM implementation (converted from dtrsm.f)
static
void trsm_ref(char side, char uplo, char transa, char diag, int m, int n,
               real alpha, const real *A, int lda, real *B, int ldb) {
  int i, j, k;
  int lside = side == 'L' || side == 'l';
  int nounit = diag == 'N' || diag == 'n';
  int upper = uplo == 'U' || uplo == 'u';
  int trans = transa == 'T' || transa == 't' || transa == 'C' || transa == 'c';
  if (m == 0 || n == 0)
    return;

  if (alpha == 0.0) {
    for (j = 0; j < n; ++j) {
      for (i = 0; i < m; ++i) {
        B[i + j * ldb] = 0.0;
      }
    }
    return;
  }

  if (lside) {
    if (!trans) {
      // SIDE = 'L', TRANSA = 'N'
      if (upper) {
        // UPLO = 'U'
        for (j = 0; j < n; ++j) {
          if (alpha != 1.0) {
            for (i = 0; i < m; ++i)
              B[i + j * ldb] *= alpha;
          }
          for (k = m - 1; k >= 0; --k) {
            if (B[k + j * ldb] != 0.0) {
              if (nounit)
                B[k + j * ldb] /= A[k + k * lda];
              for (i = 0; i < k; ++i) {
                B[i + j * ldb] -= B[k + j * ldb] * A[i + k * lda];
              }
            }
          }
        }
      } else {
        // UPLO = 'L'
        for (j = 0; j < n; ++j) {
          if (alpha != 1.0) {
            for (i = 0; i < m; ++i)
              B[i + j * ldb] *= alpha;
          }
          for (k = 0; k < m; ++k) {
            if (B[k + j * ldb] != 0.0) {
              if (nounit)
                B[k + j * ldb] /= A[k + k * lda];
              for (i = k + 1; i < m; ++i) {
                B[i + j * ldb] -= B[k + j * ldb] * A[i + k * lda];
              }
            }
          }
        }
      }
    } else {
      // SIDE = 'L', TRANSA = 'T'
      if (upper) {
        // UPLO = 'U'
        for (j = 0; j < n; ++j) {
          for (i = 0; i < m; ++i) {
            real temp = alpha * B[i + j * ldb];
            for (k = 0; k < i; ++k) {
              temp -= A[k + i * lda] * B[k + j * ldb];
            }
            if (nounit)
              temp /= A[i + i * lda];
            B[i + j * ldb] = temp;
          }
        }
      } else {
        // UPLO = 'L'
        for (j = 0; j < n; ++j) {
          for (i = m - 1; i >= 0; --i) {
            real temp = alpha * B[i + j * ldb];
            for (k = i + 1; k < m; ++k) {
              temp -= A[k + i * lda] * B[k + j * ldb];
            }
            if (nounit)
              temp /= A[i + i * lda];
            B[i + j * ldb] = temp;
          }
        }
      }
    }
  } else {
    // SIDE = 'R'
    if (!trans) {
      // SIDE = 'R', TRANSA = 'N'
      if (upper) {
        // UPLO = 'U'
        for (j = 0; j < n; ++j) {
          if (alpha != 1.0) {
            for (i = 0; i < m; ++i)
              B[i + j * ldb] *= alpha;
          }
          for (k = 0; k < j; ++k) {
            if (A[k + j * lda] != 0.0) {
              for (i = 0; i < m; ++i) {
                B[i + j * ldb] -= A[k + j * lda] * B[i + k * ldb];
              }
            }
          }
          if (nounit) {
            real temp = 1.0 / A[j + j * lda];
            for (i = 0; i < m; ++i)
              B[i + j * ldb] *= temp;
          }
        }
      } else {
        // UPLO = 'L'
        for (j = n - 1; j >= 0; --j) {
          if (alpha != 1.0) {
            for (i = 0; i < m; ++i)
              B[i + j * ldb] *= alpha;
          }
          for (k = j + 1; k < n; ++k) {
            if (A[k + j * lda] != 0.0) {
              for (i = 0; i < m; ++i) {
                B[i + j * ldb] -= A[k + j * lda] * B[i + k * ldb];
              }
            }
          }
          if (nounit) {
            real temp = 1.0 / A[j + j * lda];
            for (i = 0; i < m; ++i)
              B[i + j * ldb] *= temp;
          }
        }
      }
    } else {
      // SIDE = 'R', TRANSA = 'T'
      if (upper) {
        // UPLO = 'U'
        for (k = n - 1; k >= 0; --k) {
          if (nounit) {
            real temp = 1.0 / A[k + k * lda];
            for (i = 0; i < m; ++i)
              B[i + k * ldb] *= temp;
          }
          for (j = 0; j < k; ++j) {
            if (A[j + k * lda] != 0.0) {
              real temp = A[j + k * lda];
              for (i = 0; i < m; ++i) {
                B[i + j * ldb] -= temp * B[i + k * ldb];
              }
            }
          }
          if (alpha != 1.0) {
            for (i = 0; i < m; ++i)
              B[i + k * ldb] *= alpha;
          }
        }
      } else {
        // UPLO = 'L'
        for (k = 0; k < n; ++k) {
          if (nounit) {
            real temp = 1.0 / A[k + k * lda];
            for (i = 0; i < m; ++i)
              B[i + k * ldb] *= temp;
          }
          for (j = k + 1; j < n; ++j) {
            if (A[j + k * lda] != 0.0) {
              real temp = A[j + k * lda];
              for (i = 0; i < m; ++i) {
                B[i + j * ldb] -= temp * B[i + k * ldb];
              }
            }
          }
          if (alpha != 1.0) {
            for (i = 0; i < m; ++i)
              B[i + k * ldb] *= alpha;
          }
        }
      }
    }
  }
}

static
void BLASNAME(trsm_opt)(char side, char uplo, char transa, char diag, int m, int n,
              real alpha, const real *A, int lda, real *B, int ldb) {
  int side_dim = (side == 'L' || side == 'l') ? m : n;
  if (side_dim == 1) {
    real a00 = A[0];
    if (alpha != 1.0 || a00 != 1.0) {
      real scale = alpha / a00;
      #pragma omp parallel for collapse(2)
      for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
          B[i + j * ldb] *= scale;
        }
      }
    }
    return;
  }

  // Fast Path for Small Matrices
  if (m <= TRSM_BLK && n <= TRSM_BLK) {
    trsm_ref(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
    return;
  }

  // Scale B by alpha upfront to avoid real-scaling in blocked updates
  if (alpha != 1.0) {
#pragma omp parallel for collapse(2)
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < m; ++i) {
        B[i + j * ldb] *= alpha;
      }
    }
    alpha = 1.0;
  }

  int lside = (side == 'L' || side == 'l');
  int lower = (uplo == 'L' || uplo == 'l');
  int trans = (transa == 'T' || transa == 't' || transa == 'C' || transa == 'c');

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
        long dep_idx = lside ? (long)k : (long)k * ldb;

#pragma omp task depend(inout : B[dep_idx]) firstprivate(k, blk_size, alpha)
        {
          if (lside) {
            const real * Ak = &A[k + k * lda];
            real *Bk = &B[k]; // B row k
            trsm_ref(side, uplo, transa, diag, blk_size, n, alpha, Ak, lda, Bk, ldb);
          } else {
            const real *Ak = &A[k + k * lda];
            real *Bk = &B[k * ldb]; // B col k
            trsm_ref(side, uplo, transa, diag, m, blk_size, alpha, Ak, lda, Bk, ldb);
          }
        }

        // 2. Task: Update Off-Diagonal
        if (forward) {
          for (int i = k + blk_size; i < limit; i += TRSM_BLK) {
            int i_size = (i + TRSM_BLK > limit) ? limit - i : TRSM_BLK;

            long src_idx = lside ? (long)k : (long)k * ldb;
            long dst_idx = lside ? (long)i : (long)i * ldb;

#pragma omp task depend(in : B[src_idx]) depend(inout : B[dst_idx]) firstprivate(i, k, blk_size, i_size)
            {
              // GEMM Update
              if (lside) {
                // B[i] -= op(A[i,k]) * B[k]
                const real *A_ptr = trans ? &A[k + i * lda] : &A[i + k * lda];
                gemm_opt(trans ? transa : 'N', 'N', i_size, n, blk_size, -1.0,
                         A_ptr, lda, &B[k], ldb, 1.0, &B[i], ldb);
              } else {
                // B[i] -= B[k] * op(A[k,i])
                const real *A_ptr = trans ? &A[i + k * lda] : &A[k + i * lda];
                gemm_opt('N', trans ? transa : 'N', m, i_size, blk_size, -1.0,
                         &B[k * ldb], ldb, A_ptr, lda, 1.0, &B[i * ldb], ldb);
              }
            }
          }
        } else {
          // Backward Loop
          for (int temp_i = k - TRSM_BLK;; temp_i -= TRSM_BLK) {
            int i = temp_i;
            int i_size = TRSM_BLK;
            if (temp_i < 0) {
              i = 0;
              i_size = temp_i + TRSM_BLK;
            }

            if (i_size > 0) {
              long src_idx = lside ? (long)k : (long)k * ldb;
              long dst_idx = lside ? (long)i : (long)i * ldb;

#pragma omp task depend(in : B[src_idx]) depend(inout : B[dst_idx]) firstprivate(i, k, blk_size, i_size)
              {
                if (lside) {
                  const real *A_ptr = trans ? &A[k + i * lda] : &A[i + k * lda];
                  gemm_opt(transa, 'N', i_size, n, blk_size, -1.0,
                           A_ptr, lda, &B[k], ldb, 1.0, &B[i], ldb);
                } else {
                  const real *A_ptr = trans ? &A[i + k * lda] : &A[k + i * lda];
                  gemm_opt('N', transa, m, i_size, blk_size, -1.0,
                           &B[k * ldb], ldb, A_ptr, lda, 1.0, &B[i * ldb], ldb);
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




void BLASNAME(trsm)(char const *side, char const *uplo, char const *transa, char const *diag, int const *m, int const *n,
            real const *alpha, const real *restrict A, int const *lda, real *restrict B, int const *ldb) {
  BLASNAME(trsm_opt)(*side, *uplo, *transa, *diag, *m, *n, *alpha, A, *lda, B, *ldb);
  // dtrsm_ref(*side, *uplo, *transa, *diag, *m, *n, *alpha, A, *lda, B, *ldb);
}
