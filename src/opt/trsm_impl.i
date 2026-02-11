/**
 * dtrsm_omp.c
 *
 * OpenMP 4.5 Parallelized DTRSM (Double-precision Triangular Solve with
 * Multiple RHS)
 *
 * Features:
 *  - Trivial Parallelization:
 *      Side=L -> Parallelize over columns of B
 *      Side=R -> Parallelize over rows of B
 *  - Cache Blocking: Decomposes problem into DGEMM updates and small TRSM
 * solves.
 *  - Block Size: Tuned via TRSM_BLK macro (default 128).
 */

#ifndef BLASNAME
#include "d.c"
#endif


#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

// -----------------------------------------------------------------------------
// External BLAS Prototypes
// -----------------------------------------------------------------------------
/*
 * We assume a standard Fortran-compatible BLAS interface (pointers +
 * underscores). If linking against a C-interface BLAS (like cblas), adapters
 * would be needed.
 */
extern void BLASNAME(gemm)(
  const char *transa, const char *transb, const int *m,
  const int *n, const int *k, const real *alpha,
  const real *a, const int *lda, const real *b,
  const int *ldb, const real *beta, real *c,
  const int *ldc);

extern void xerbla_(const char *srname, const int *info, int len);

static const real one = 1.0;
static const real neg_one = -1.0;

// -----------------------------------------------------------------------------
// Unblocked Scalar DTRSM (The Base Case)
// -----------------------------------------------------------------------------
/*
 * Solves op(A)*X = alpha*B or X*op(A) = alpha*B for small blocks.
 * This function does not use DGEMM; it uses nested loops.
 */
static void trsm_unblocked(
    char side, char uplo, char transa, char diag,
    const int m, const int n,
    const real alpha,
    const real *a, const int lda,
    real *b, const int ldb
) {
  // Quick return
  if (m == 0 || n == 0)
    return;

  // Apply alpha scaling to B if alpha is not 1.0
  if (alpha == 0.0) {
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < m; i++) {
        b[j * ldb + i] = 0.0;
      }
    }
    return;
  }

  if (alpha != 1.0) {
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < m; i++) {
        b[j * ldb + i] *= alpha;
      }
    }
  }

  int lside = (side == 'L' || side == 'l');
  int lower = (uplo == 'L' || uplo == 'l');
  int nounit = (diag == 'N' || diag == 'n');
  int trans = (transa == 'T' || transa == 't' || transa == 'C' || transa == 'c');

  // -------------------------------------------------------------------------
  // SIDE LEFT: op(A) * X = B
  // -------------------------------------------------------------------------
  if (lside) {
    if (!trans) {
      // Case 1: A * X = B
      if (lower) {
        // Lower Triangular: Forward Substitution
        for (int j = 0; j < n; j++) {
          for (int k = 0; k < m; k++) {
            if (nounit)
              b[j * ldb + k] /= a[k * lda + k];
            for (int i = k + 1; i < m; i++) {
              b[j * ldb + i] -= b[j * ldb + k] * a[k * lda + i];
            }
          }
        }
      } else {
        // Upper Triangular: Backward Substitution
        for (int j = 0; j < n; j++) {
          for (int k = m - 1; k >= 0; k--) {
            if (nounit)
              b[j * ldb + k] /= a[k * lda + k];
            for (int i = 0; i < k; i++) {
              b[j * ldb + i] -= b[j * ldb + k] * a[k * lda + i];
            }
          }
        }
      }
    } else {
      // Case 2: A^T * X = B
      if (lower) {
        // Lower Transpose (behaves like Upper): Backward Loop
        for (int j = 0; j < n; j++) {
          for (int k = m - 1; k >= 0; k--) {
            real sum = 0.0;
            for (int i = k + 1; i < m; i++) {
              sum += a[k * lda + i] * b[j * ldb + i];
            }
            b[j * ldb + k] -= sum;
            if (nounit)
              b[j * ldb + k] /= a[k * lda + k];
          }
        }
      } else {
        // Upper Transpose (behaves like Lower): Forward Loop
        for (int j = 0; j < n; j++) {
          for (int k = 0; k < m; k++) {
            real sum = 0.0;
            for (int i = 0; i < k; i++) {
              sum += a[k * lda + i] * b[j * ldb + i];
            }
            b[j * ldb + k] -= sum;
            if (nounit)
              b[j * ldb + k] /= a[k * lda + k];
          }
        }
      }
    }
  }
  // -------------------------------------------------------------------------
  // SIDE RIGHT: X * op(A) = B
  // -------------------------------------------------------------------------
  else {
    if (!trans) {
      // Case 3: X * A = B
      if (lower) {
        // Lower Triangular: Backward Loop (solve X_n, then X_{n-1}...)
        for (int j = n - 1; j >= 0; j--) {
          if (nounit) {
            real invDiag = 1.0 / a[j * lda + j];
            for (int i = 0; i < m; i++)
              b[j * ldb + i] *= invDiag;
          }
          for (int k = 0; k < j; k++) {
            if (a[k * lda + j] != 0.0) {
              real Akj = a[k * lda + j];
              for (int i = 0; i < m; i++) {
                b[k * ldb + i] -= Akj * b[j * ldb + i];
              }
            }
          }
        }
      } else {
        // Upper Triangular: Forward Loop
        for (int j = 0; j < n; j++) {
          if (nounit) {
            real invDiag = 1.0 / a[j * lda + j];
            for (int i = 0; i < m; i++)
              b[j * ldb + i] *= invDiag;
          }
          for (int k = j + 1; k < n; k++) {
            if (a[k * lda + j] != 0.0) {
              real Akj = a[k * lda + j];
              for (int i = 0; i < m; i++) {
                b[k * ldb + i] -= Akj * b[j * ldb + i];
              }
            }
          }
        }
      }
    } else {
      // Case 4: X * A^T = B
      if (lower) {
        // Lower Transpose: Forward Loop
        for (int k = 0; k < n; k++) {
          if (nounit) {
            real invDiag = 1.0 / a[k * lda + k];
            for (int i = 0; i < m; i++)
              b[k * ldb + i] *= invDiag;
          }
          for (int j = k + 1; j < n; j++) {
            if (a[k * lda + j] != 0.0) {
              real Ajk = a[k * lda + j]; // A[j,k] in C
              for (int i = 0; i < m; i++) {
                b[j * ldb + i] -= Ajk * b[k * ldb + i];
              }
            }
          }
        }
      } else {
        // Upper Transpose: Backward Loop
        for (int k = n - 1; k >= 0; k--) {
          if (nounit) {
            real invDiag = 1.0 / a[k * lda + k];
            for (int i = 0; i < m; i++)
              b[k * ldb + i] *= invDiag;
          }
          for (int j = 0; j < k; j++) {
            if (a[k * lda + j] != 0.0) {
              real Ajk = a[k * lda + j];
              for (int i = 0; i < m; i++) {
                b[j * ldb + i] -= Ajk * b[k * ldb + i];
              }
            }
          }
        }
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Blocked DTRSM (Sequential Logic)
// -----------------------------------------------------------------------------
/*
 * This function is called by each OpenMP thread on its assigned panel of B.
 * It iterates through diagonal blocks of A, calls the scalar TRSM for the
 * diagonal block, and uses DGEMM for updates.
 */
static void trsm_blocked(
    char side, char uplo, char transa, char diag,
    const int m, const int n,
    const real alpha,
    const real *a, const int lda,
    real *b, const int ldb
) {
  // Constants for DGEMM

  int lside = (side == 'L' || side == 'l');
  int lower = (uplo == 'L' || uplo == 'l');
  int trans = (transa == 'T' || transa == 't' || transa == 'C' || transa == 'c');

  // -------------------------------------------------------------------------
  // SIDE LEFT
  // -------------------------------------------------------------------------
  if (lside) {
    if (!trans) {
      // A * X = B
      if (lower) {
        // Left, Lower, NoTrans: Forward Blocked Loop
        for (int k = 0; k < m; k += TRSM_BLK) {
          int jb = MIN(TRSM_BLK, m - k);
          trsm_unblocked('L', 'L', 'N', diag, jb, n, alpha, &a[k * lda + k],
                          lda, &b[0 * ldb + k], ldb);

          if (k + jb < m) {
            int m_rest = m - (k + jb);
            gemm_opt('N', 'N', m_rest, n, jb, neg_one, &a[k * lda + (k + jb)],
                   lda, &b[0 * ldb + k], ldb, one,
                   &b[0 * ldb + (k + jb)], ldb);
          }
        }
      } else {
        // Left, Upper, NoTrans: Backward Blocked Loop
        for (int k = m; k > 0; k -= TRSM_BLK) {
          int jb = MIN(TRSM_BLK, k);
          int k_start = k - jb;
          trsm_unblocked('L', 'U', 'N', diag, jb, n, alpha,
                          &a[k_start * lda + k_start], lda,
                          &b[0 * ldb + k_start], ldb);

          if (k_start > 0) {
            gemm_opt('N', 'N', k_start, n, jb, neg_one, &a[k_start * lda + 0],
                   lda, &b[0 * ldb + k_start], ldb, one, &b[0 * ldb + 0], ldb);
          }
        }
      }
    } else {
      // A^T * X = B
      if (lower) {
        // Left, Lower, Trans: Backward Loop (Behaves like Upper)
        for (int k = m; k > 0; k -= TRSM_BLK) {
          int jb = MIN(TRSM_BLK, k);
          int k_start = k - jb;
          trsm_unblocked('L', 'L', 'T', diag, jb, n, alpha,
                          &a[k_start * lda + k_start], lda,
                          &b[0 * ldb + k_start], ldb);

          if (k_start > 0) {
            gemm_opt('T', 'N', k_start, n, jb, neg_one, &a[0 * lda + k_start],
                   lda, &b[0 * ldb + k_start], ldb, one,
                   &b[0 * ldb + 0], ldb);
          }
        }
      } else {
        // Left, Upper, Trans: Forward Loop (Behaves like Lower)
        for (int k = 0; k < m; k += TRSM_BLK) {
          int jb = MIN(TRSM_BLK, m - k);
          trsm_unblocked('L', 'U', 'T', diag, jb, n, alpha, &a[k * lda + k],
                          lda, &b[0 * ldb + k], ldb);

          if (k + jb < m) {
            int m_rest = m - (k + jb);
            gemm_opt('T', 'N', m_rest, n, jb, neg_one, &a[(k + jb) * lda + k],
                   lda, &b[0 * ldb + k], ldb, one,
                   &b[0 * ldb + (k + jb)], ldb);
          }
        }
      }
    }
  }
  // -------------------------------------------------------------------------
  // SIDE RIGHT
  // -------------------------------------------------------------------------
  else {
    if (!trans) {
      // X * A = B
      if (lower) {
        // Right, Lower, NoTrans: Backward Loop
        for (int k = n; k > 0; k -= TRSM_BLK) {
          int jb = MIN(TRSM_BLK, k);
          int k_start = k - jb;
          trsm_unblocked('R', 'L', 'N', diag, m, jb, alpha,
                          &a[k_start * lda + k_start], lda,
                          &b[k_start * ldb + 0], ldb);

          if (k_start > 0) {
            gemm_opt('N', 'N', m, k_start, jb, neg_one, &b[k_start * ldb + 0],
                   ldb, &a[0 * lda + k_start], lda, one,
                   &b[0 * ldb + 0], ldb);
          }
        }
      } else {
        // Right, Upper, NoTrans: Forward Loop
        for (int k = 0; k < n; k += TRSM_BLK) {
          int jb = MIN(TRSM_BLK, n - k);
          trsm_unblocked('R', 'U', 'N', diag, m, jb, alpha, &a[k * lda + k],
                          lda, &b[k * ldb + 0], ldb);

          if (k + jb < n) {
            int n_rest = n - (k + jb);
            gemm_opt('N', 'N', m, n_rest, jb, neg_one, &b[k * ldb + 0], ldb,
                   &a[(k + jb) * lda + k], lda, one,
                   &b[(k + jb) * ldb + 0], ldb);
          }
        }
      }
    } else {
      // X * A^T = B
      if (lower) {
        // Right, Lower, Trans: Forward Loop
        for (int k = 0; k < n; k += TRSM_BLK) {
          int jb = MIN(TRSM_BLK, n - k);
          trsm_unblocked('R', 'L', 'T', diag, m, jb, alpha, &a[k * lda + k],
                          lda, &b[k * ldb + 0], ldb);

          if (k + jb < n) {
            int n_rest = n - (k + jb);
            gemm_opt('N', 'T', m, n_rest, jb, neg_one, &b[k * ldb + 0],
                   ldb, &a[k * lda + (k + jb)], lda, one,
                   &b[(k + jb) * ldb + 0], ldb);
          }
        }
      } else {
        // Right, Upper, Trans: Backward Loop
        for (int k = n; k > 0; k -= TRSM_BLK) {
          int jb = MIN(TRSM_BLK, k);
          int k_start = k - jb;
          trsm_unblocked('R', 'U', 'T', diag, m, jb, alpha,
                          &a[k_start * lda + k_start], lda,
                          &b[k_start * ldb + 0], ldb);

          if (k_start > 0) {
            gemm_opt('N', 'T', m, k_start, jb, neg_one,
                   &b[k_start * ldb + 0], ldb, &a[k_start * lda + 0], lda,
                   one, &b[0 * ldb + 0], ldb);
          }
        }
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Parallel Driver (The Entry Point)
// -----------------------------------------------------------------------------
void BLASNAME(trsm)(const char *side, const char *uplo, const char *transa,
           const char *diag, const int *m, const int *n, const real *alpha,
           const real *a, const int *lda, real *b, const int *ldb) {
  // 1. Input Validation
  int info = 0;
  char s = *side;
  char u = *uplo;
  char t = *transa;
  char d = *diag;

  int lside = (s == 'L' || s == 'l');
  int nrowa = lside ? *m : *n;

  if (!lside &&!(*side == 'R' || *side == 'r'))
    info = 1;
  else if (!(*uplo == 'U' || *uplo == 'u') && !(*uplo == 'L' || *uplo == 'l'))
    info = 2;
  else if (!(*transa == 'N' || *transa == 'n') && !(*transa == 'T' || *transa == 't') &&
           !(*transa == 'C' || *transa == 'c'))
    info = 3;
  else if (!(*diag == 'U' || *diag == 'u') && !(*diag == 'N' || *diag == 'n'))
    info = 4;
  else if (*m < 0)
    info = 5;
  else if (*n < 0)
    info = 6;
  else if (*lda < MAX(1, nrowa))
    info = 9;
  else if (*ldb < MAX(1, *m))
    info = 11;

  if (info != 0) {
    xerbla_("DTRSM ", &info, 5);
    return;
  }

  // Quick return
  if (*m == 0 || *n == 0)
    return;

  // 2. Alpha Scaling
  if (*alpha == 0.0) {
#pragma omp parallel for schedule(static)
    for (int j = 0; j < *n; j++) {
      for (int i = 0; i < *m; i++) {
        b[j * (*ldb) + i] = 0.0;
      }
    }
    return;
  }

  if (*alpha != 1.0) {
#pragma omp parallel for schedule(static)
    for (int j = 0; j < *n; j++) {
      for (int i = 0; i < *m; i++) {
        b[j * (*ldb) + i] *= (*alpha);
      }
    }
  }

  // Use 1.0 for the internal algorithms since B is pre-scaled
  real internal_alpha = 1.0;

  int nthread = omp_get_max_threads();
  // 3. Trivial Parallelization Strategy
  if (lside) {
// ---------------------------------------------------------
// SIDE LEFT: op(A) * X = B
// Parallelize over COLUMNS of B (index j).
// ---------------------------------------------------------

    int block_size = (*n + nthread - 1) / nthread;

#pragma omp parallel for schedule(static)
    for (int ithread = 0; ithread < nthread; ++ithread) {
      int j = ithread * block_size;
      if (j >= *n) continue;
      int jend = MIN((ithread + 1) * block_size, *n);
      int jb = jend - j;

      real *b_panel = &b[j * (*ldb)];

      trsm_blocked(s, u, t, d, *m, jb, internal_alpha, a, *lda,
                    b_panel, *ldb);
    }
  } else {
    int block_size = (*m + nthread - 1) / nthread;

// ---------------------------------------------------------
// SIDE RIGHT: X * op(A) = B
// Parallelize over ROWS of B (index i).
// ---------------------------------------------------------
#pragma omp parallel for schedule(static)
    for (int ithread = 0; ithread < nthread; ++ithread) {
      int i = ithread * block_size;
      if (i >= *m) continue;
      int iend = MIN((ithread+1)*block_size, *m);
      int ib = iend - i;
      real *b_panel = &b[i];

      trsm_blocked(s, u, t, d, ib, *n, internal_alpha, a, *lda, b_panel, *ldb);
    }
  }
}


// void BLASNAME(trsm)(char const *side, char const *uplo, char const *transa, char const *diag, int const *m, int const *n,
//             real const *alpha, const real *restrict A, int const *lda, real *restrict B, int const *ldb) {
//   BLASNAME(trsm_opt)(*side, *uplo, *transa, *diag, *m, *n, *alpha, A, *lda, B, *ldb);
//   // dtrsm_ref(*side, *uplo, *transa, *diag, *m, *n, *alpha, A, *lda, B, *ldb);
// }




