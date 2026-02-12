/*
 * Double-Double GEMM Implementation using AVX2 and OpenMP
 * Target: x86-64 Haswell/Skylake or newer
 * Compile with: g++ -O3 -mavx2 -mfma -fopenmp gemm_dd.cpp -o gemm_dd
 */

#include "Float64x2.hh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <omp.h>
#include <vector>

// ============================================================================
// Data Structures and Constants
// ============================================================================

// Cache Blocking Parameters (Tuned for AVX2 Double-Double)
// DD is 16 bytes. L1=32KB, L2=256KB typically.
// MC * KC * 16 bytes should fit in L2 (approx).
// KC * NC * 16 bytes should fit in L3.
#define MC 64
#define KC 128
#define NC 256

// Register Blocking Parameters
// MR: Rows of C in registers (vector length 4 doubles * 1 vector = 4 rows)
// NR: Cols of C in registers
#define MR 4
#define NR 1

// ============================================================================
// AVX2 Double-Double Arithmetic Helpers
// ============================================================================

// QuickTwoSum: s = a + b, e = b - (s - a)
// Requirement: |a| >= |b|
static inline void avx2_quick_two_sum(__m256d a, __m256d b, __m256d &s,
                                      __m256d &e) {
  s = _mm256_add_pd(a, b);
  e = _mm256_sub_pd(b, _mm256_sub_pd(s, a));
}

// TwoSum: s = a + b, e = (a - (s - v)) + (b - v)
static inline void avx2_two_sum(__m256d a, __m256d b, __m256d &s, __m256d &e) {
  s = _mm256_add_pd(a, b);
  __m256d v = _mm256_sub_pd(s, a);
  e = _mm256_add_pd(_mm256_sub_pd(a, _mm256_sub_pd(s, v)), _mm256_sub_pd(b, v));
}

// TwoProd: p = a * b, e = a * b - p (using FMA)
static inline void avx2_two_prod(__m256d a, __m256d b, __m256d &p, __m256d &e) {
  p = _mm256_mul_pd(a, b);
  e = _mm256_fmsub_pd(a, b, p); // FMA: a*b - p
}

// Double-Double Addition: C = A + B
// A = (ahi, alo), B = (bhi, blo), C = (chi, clo)
static inline void avx2_add_dd(__m256d ahi, __m256d alo, __m256d bhi,
                               __m256d blo, __m256d &chi, __m256d &clo) {
  __m256d s, t1, t2, t3;
  // 1. TwoSum(ahi, bhi) -> (s, t1)
  avx2_two_sum(ahi, bhi, s, t1);
  // 2. t2 = alo + blo
  t2 = _mm256_add_pd(alo, blo);
  // 3. t3 = t1 + t2
  t3 = _mm256_add_pd(t1, t2);
  // 4. QuickTwoSum(s, t3) -> (chi, clo)
  avx2_quick_two_sum(s, t3, chi, clo);
}

// Double-Double Multiplication: C = A * B
// Discards O(eps^2) terms
static inline void avx2_mul_dd(__m256d ahi, __m256d alo, __m256d bhi,
                               __m256d blo, __m256d &chi, __m256d &clo) {
  __m256d p1, p2, p3, t;
  // 1. TwoProd(ahi, bhi) -> (p1, p2)
  avx2_two_prod(ahi, bhi, p1, p2);
  // 2. t = ahi*blo + alo*bhi (Cross terms)
  t = _mm256_fmadd_pd(ahi, blo, _mm256_mul_pd(alo, bhi));
  // 3. p3 = p2 + t
  p3 = _mm256_add_pd(p2, t);
  // 4. QuickTwoSum(p1, p3) -> (chi, clo)
  avx2_quick_two_sum(p1, p3, chi, clo);
}

// ============================================================================
// Scalar Double-Double Arithmetic Helpers (for Packing/Scaling)
// ============================================================================

static inline void scalar_two_prod(double a, double b, double &p, double &e) {
  p = a * b;
  e = std::fma(a, b, -p);
}

static inline void scalar_quick_two_sum(double a, double b, double &s,
                                        double &e) {
  s = a + b;
  e = b - (s - a);
}

static inline void scalar_two_sum(double a, double b, double &s, double &e) {
  s = a + b;
  double v = s - a;
  e = (a - (s - v)) + (b - v);
}

static inline void scalar_mul_dd(double ahi, double alo, double bhi, double blo,
                                 double &chi, double &clo) {
  double p1, p2;
  scalar_two_prod(ahi, bhi, p1, p2);
  double t = ahi * blo + alo * bhi;
  double p3 = p2 + t;
  scalar_quick_two_sum(p1, p3, chi, clo);
}

static inline void scalar_add_dd(double ahi, double alo, double bhi, double blo,
                                 double &chi, double &clo) {
  double s, t1, t2, t3;
  scalar_two_sum(ahi, bhi, s, t1);
  t2 = alo + blo;
  t3 = t1 + t2;
  scalar_quick_two_sum(s, t3, chi, clo);
}

// ============================================================================
// Packing Kernels (AoS -> SoA)
// ============================================================================

// Pack A: Extract MR x KC block, transpose to SoA layout
// Output Layout: Contiguous vectors of A_hi, then A_lo for MR rows.
// For MR=4, each load fetches 4 structs (hi,lo pairs).
void pack_A(int k, const float64x2 *A, int incRowA, int incColA, double *buffer,
            int mr) {
  // We assume MR=4. We process 4 rows of A at a time.
  // Input A is [ (h0,l0), (h1,l1), (h2,l2), (h3,l3) ] in memory.
  // We want registers: A_hi = [h0,h1,h2,h3], A_lo = [l0,l1,l2,l3]

  if (incRowA == 1 && mr == 4) {
    // Fast path for Column Major (incRow=1)
    for (int p = 0; p < k; ++p) {
      const double *ptr = (const double *)(A + p * incColA);
      __m256d r0 = _mm256_loadu_pd(ptr);
      __m256d r1 = _mm256_loadu_pd(ptr + 4);

      __m256d t1 = _mm256_permute4x64_pd(r0, _MM_SHUFFLE(3, 1, 2, 0));
      __m256d t2 = _mm256_permute4x64_pd(r1, _MM_SHUFFLE(3, 1, 2, 0));

      __m256d a_hi = _mm256_permute2f128_pd(t1, t2, 0x20);
      __m256d a_lo = _mm256_permute2f128_pd(t1, t2, 0x31);

      _mm256_storeu_pd(buffer, a_hi);
      _mm256_storeu_pd(buffer + 4, a_lo);
      buffer += 8;
    }
  } else {
    // General path
    for (int p = 0; p < k; ++p) {
      const float64x2 *col_ptr = A + p * incColA;
      double h[4] = {0}, l[4] = {0}; // Initialize with 0 for padding
      for (int r = 0; r < mr; ++r) {
        const float64x2 *elem = col_ptr + r * incRowA;
        h[r] = elem->limbs[0];
        l[r] = elem->limbs[1];
      }
      _mm256_storeu_pd(buffer, _mm256_loadu_pd(h));
      _mm256_storeu_pd(buffer + 4, _mm256_loadu_pd(l));
      buffer += 8;
    }
  }
}

// Pack B: Extract KC x NR block.
// Since NR=1, we just pack scalars for broadcast.
// Stored as pairs [hi, lo], [hi, lo]... sequentially
void pack_B(int k, const float64x2 *B, int incRowB, double *buffer,
            const float64x2 &alpha) {
  bool is_unit = (alpha.limbs[0] == 1.0 && alpha.limbs[1] == 0.0);
  for (int p = 0; p < k; ++p) {
    // Just copy the struct as is.
    // B is accessed as scalars in the kernel and broadcasted.
    const float64x2 *ptr = B + p * incRowB;
    double bhi = ptr->limbs[0];
    double blo = ptr->limbs[1];
    if (!is_unit) {
      scalar_mul_dd(bhi, blo, alpha.limbs[0], alpha.limbs[1], bhi, blo);
    }
    buffer[0] = bhi;
    buffer[1] = blo;
    buffer += 2;
  }
}

// ============================================================================
// Micro-Kernel
// ============================================================================

// Compute 4x1 block of C.
// Registers:
//   Accumulators: c_hi, c_lo (2 registers)
//   A: a_hi, a_lo (2 registers)
//   B: b_hi, b_lo (2 registers)
//   Temps: 6 registers available
static void micro_kernel(int k, const double *packA, const double *packB,
                         float64x2 *C, int ldc) {

  // Initialize Accumulators
  __m256d c_hi = _mm256_setzero_pd();
  __m256d c_lo = _mm256_setzero_pd();

  for (int p = 0; p < k; ++p) {
    // Load A (SoA packed)
    __m256d a_hi = _mm256_loadu_pd(packA);
    __m256d a_lo = _mm256_loadu_pd(packA + 4);
    packA += 8;

    // Load B (Scalar broadcast)
    // packB points to [b_hi, b_lo]
    __m256d b_hi = _mm256_broadcast_sd(packB);
    __m256d b_lo = _mm256_broadcast_sd(packB + 1);
    packB += 2;

    // Complex FMA: C += A * B

    // 1. Prod = A * B
    __m256d p_hi, p_lo;
    avx2_mul_dd(a_hi, a_lo, b_hi, b_lo, p_hi, p_lo);

    // 2. Sum = C + Prod
    avx2_add_dd(c_hi, c_lo, p_hi, p_lo, c_hi, c_lo);
  }

  // Update C in memory (AoS)
  // C is [ (h0,l0), (h1,l1), (h2,l2), (h3,l3) ]
  // Registers are SoA: c_hi=[h0..h3], c_lo=[l0..l3]

  // 1. Load existing C
  // We need to load 4 structs like in PackA
  double *c_ptr = (double *)C;
  __m256d r0 = _mm256_loadu_pd(c_ptr);     // h0 l0 h1 l1
  __m256d r1 = _mm256_loadu_pd(c_ptr + 4); // h2 l2 h3 l3

  // De-interleave C
  __m256d t1 = _mm256_permute4x64_pd(r0, _MM_SHUFFLE(3, 1, 2, 0));
  __m256d t2 = _mm256_permute4x64_pd(r1, _MM_SHUFFLE(3, 1, 2, 0));
  __m256d old_c_hi = _mm256_permute2f128_pd(t1, t2, 0x20);
  __m256d old_c_lo = _mm256_permute2f128_pd(t1, t2, 0x31);

  // 2. Add computed delta to C
  avx2_add_dd(old_c_hi, old_c_lo, c_hi, c_lo, c_hi, c_lo);

  // 3. Interleave back to AoS for storage
  // c_hi = [h0, h1, h2, h3]
  // c_lo = [l0, l1, l2, l3]

  // unpacklo: [h0, l0, h2, l2]
  __m256d res0 = _mm256_unpacklo_pd(c_hi, c_lo);
  // unpackhi: [h1, l1, h3, l3]
  __m256d res1 = _mm256_unpackhi_pd(c_hi, c_lo);

  // Permute 128-bit lanes to get correct order:
  // final0: [h0, l0, h1, l1]
  __m256d final0 = _mm256_permute2f128_pd(res0, res1, 0x20);
  // final1: [h2, l2, h3, l3]
  __m256d final1 = _mm256_permute2f128_pd(res0, res1, 0x31);

  _mm256_storeu_pd(c_ptr, final0);
  _mm256_storeu_pd(c_ptr + 4, final1);
}

// ============================================================================
// Macro-Kernel and Tiling
// ============================================================================

void gemm_opt(int M, int N, int K, float64x2 alpha, const float64x2 *A,
              int incRowA, int incColA, const float64x2 *B, int incRowB,
              int incColB, float64x2 beta, float64x2 *C, int ldc) {

  // Beta Scaling
  if (beta.limbs[0] == 0.0 && beta.limbs[1] == 0.0) {
#pragma omp parallel for collapse(2)
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < M; ++i) {
        C[j * ldc + i].limbs[0] = 0.0;
        C[j * ldc + i].limbs[1] = 0.0;
      }
    }
  } else if (beta.limbs[0] != 1.0 || beta.limbs[1] != 0.0) {
#pragma omp parallel for collapse(2)
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < M; ++i) {
        double chi = C[j * ldc + i].limbs[0];
        double clo = C[j * ldc + i].limbs[1];
        scalar_mul_dd(chi, clo, beta.limbs[0], beta.limbs[1], chi, clo);
        C[j * ldc + i].limbs[0] = chi;
        C[j * ldc + i].limbs[1] = clo;
      }
    }
  }

  if (alpha.limbs[0] == 0.0 && alpha.limbs[1] == 0.0)
    return;

// Parallelize Loop 5 (jc)
#pragma omp parallel
  {
    // Thread-local packing buffers
    // Pack A needs: MC * KC * sizeof(float64x2)
    std::vector<double> packA_buf(MC * KC * 2);
    // Pack B needs: KC * NC * sizeof(float64x2)
    // Since NR=1, we pack strips of B.
    // Actually, for Loop 4, we pack B once per KC block.
    // Let's use a simpler approach: Pack B inside the jc loop.
    std::vector<double> packB_buf(KC * NC * 2);

#pragma omp for collapse(2) schedule(dynamic)
    for (int jc = 0; jc < N; jc += NC) {
      for (int ic = 0; ic < M; ic += MC) {
        for (int kc = 0; kc < K; kc += KC) {
          int nc = std::min(NC, N - jc);
          int mc = std::min(MC, M - ic);
          int k_block = std::min(KC, K - kc);

          // Pack B: Block B(kc:kc+k_block, jc:jc+nc)
          // We pack column by column for B?
          // Our microkernel iterates K, so B should be packed such that
          // K dimension is contiguous for each column, or NR strip.
          // Since NR=1, we just pack columns.
          double *pB = packB_buf.data();
          for (int j = 0; j < nc; ++j) {
            pack_B(k_block, &B[(jc + j) * incColB + kc * incRowB], incRowB,
                   pB + j * k_block * 2, alpha);
          }

          // Pack A: Block A(ic:ic+mc, kc:kc+k_block)
          // We pack MR strips.
          double *pA = packA_buf.data();
          for (int i = 0; i < mc; i += MR) {
            int cur_mr = std::min(MR, mc - i);
            pack_A(k_block, &A[(ic + i) * incRowA + kc * incColA], incRowA,
                   incColA, pA + i * k_block * 2, cur_mr);
          }

          // Macro Kernel: Iterate over packed buffers
          for (int jr = 0; jr < nc; jr += NR) {   // NR=1
            for (int ir = 0; ir < mc; ir += MR) { // MR=4
              if (ir + MR <= mc) {
                micro_kernel(k_block, packA_buf.data() + ir * k_block * 2,
                             packB_buf.data() + jr * k_block * 2,
                             &C[(jc + jr) * ldc + (ic + ir)], ldc);
              } else {
                // Edge case: Process remaining rows scalar-wise
                double *pB_col = packB_buf.data() + (size_t)jr * k_block * 2;
                double *pA_strip = packA_buf.data() + (size_t)ir * k_block * 2;
                for (int r = 0; r < mc - ir; ++r) {
                  int row = ic + ir + r;
                  int col = jc + jr;
                  double chi = C[col * ldc + row].limbs[0];
                  double clo = C[col * ldc + row].limbs[1];

                  for (int p = 0; p < k_block; ++p) {
                    double bhi = pB_col[2 * p];
                    double blo = pB_col[2 * p + 1];

                    // Read from packed A (stride is 8 doubles: 4 hi + 4 lo)
                    // Layout: [h0 h1 h2 h3 l0 l1 l2 l3]
                    double ahi = pA_strip[p * 8 + r];
                    double alo = pA_strip[p * 8 + 4 + r];

                    double phi, plo;
                    scalar_mul_dd(ahi, alo, bhi, blo, phi, plo);
                    scalar_add_dd(chi, clo, phi, plo, chi, clo);
                  }
                  C[col * ldc + row].limbs[0] = chi;
                  C[col * ldc + row].limbs[1] = clo;
                }
              }
            }
          }
        }
      }
    }
  }
}


extern "C" int xerbla_(const char *, const int *, int);

extern "C" void ddgemm_(const char *transA, const char *transB, const int *m,
                        const int *n, const int *k, const float64x2 *alpha,
                        const float64x2 *A, const int *lda, const float64x2 *B,
                        const int *ldb, const float64x2 *beta, float64x2 *C,
                        const int *ldc) {
    bool nota = (*transA == 'N' || *transA == 'n');
    bool notb = (*transB == 'N' || *transB == 'n');
    int info = 0;

    if (!nota && *transA != 'T' && *transA != 't' && *transA != 'C' && *transA != 'c') {
        info = 1;
    } else if (!notb && *transB != 'T' && *transB != 't' && *transB != 'C' && *transB != 'c') {
        info = 2;
    } else if (*m < 0) {
        info = 3;
    } else if (*n < 0) {
        info = 4;
    } else if (*k < 0) {
        info = 5;
    } else if (*lda < std::max(1, nota ? *m : *k)) {
        info = 8;
    } else if (*ldb < std::max(1, notb ? *k : *n)) {
        info = 10;
    } else if (*ldc < std::max(1, *m)) {
        info = 13;
    }

    if (info != 0) {
        xerbla_("DDGEMM", &info, 6);
        return;
    }

    // Quick return if possible
    if (*m <= 0 || *n <= 0) {
        return;
    }

    int incRowA, incColA;
    if (nota) {
        incRowA = 1;
        incColA = *lda;
    } else {
        incRowA = *lda;
        incColA = 1;
    }

    int incRowB, incColB;
    if (notb) {
        incRowB = 1;
        incColB = *ldb;
    } else {
        incRowB = *ldb;
        incColB = 1;
    }

    gemm_opt(*m, *n, *k, *alpha, A, incRowA, incColA, B, incRowB, incColB, *beta,
             C, *ldc);
}
