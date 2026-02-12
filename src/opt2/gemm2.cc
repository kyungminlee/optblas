/*
 * Double-Double GEMM Implementation using AVX2 and OpenMP
 * Target: x86-64 Haswell/Skylake or newer
 * Compile with: g++ -O3 -mavx2 -mfma -fopenmp gemm_dd.cpp -o gemm_dd
 */

#include <immintrin.h>
#include <vector>
#include <omp.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>

// ============================================================================
// Data Structures and Constants
// ============================================================================

struct float62x2 {
    double limbs[2]; // limbs[0] = hi, limbs[1] = lo
};

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
static inline void avx2_quick_two_sum(__m256d a, __m256d b, __m256d &s, __m256d &e) {
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
static inline void avx2_add_dd(__m256d ahi, __m256d alo, 
                               __m256d bhi, __m256d blo, 
                               __m256d &chi, __m256d &clo) {
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
static inline void avx2_mul_dd(__m256d ahi, __m256d alo, 
                               __m256d bhi, __m256d blo, 
                               __m256d &chi, __m256d &clo) {
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
// Packing Kernels (AoS -> SoA)
// ============================================================================

// Pack A: Extract MR x KC block, transpose to SoA layout
// Output Layout: Contiguous vectors of A_hi, then A_lo for MR rows.
// For MR=4, each load fetches 4 structs (hi,lo pairs).
void pack_A(int k, const float62x2 *A, int lda, double *buffer) {
    // We assume MR=4. We process 4 rows of A at a time.
    // Input A is [ (h0,l0), (h1,l1), (h2,l2), (h3,l3) ] in memory.
    // We want registers: A_hi = [h0,h1,h2,h3], A_lo = [l0,l1,l2,l3]
    
    for (int p = 0; p < k; ++p) {
        // Load 4 structs (8 doubles)
        // r0 = [h0, l0, h1, l1]
        // r1 = [h2, l2, h3, l3]
        const double* ptr = (const double*)(A + p * lda);
        __m256d r0 = _mm256_loadu_pd(ptr); 
        __m256d r1 = _mm256_loadu_pd(ptr + 4); 

        // Permute to de-interleave
        // t1 = [h0, h1, l0, l1]
        // t2 = [h2, h3, l2, l3]
        __m256d t1 = _mm256_permute4x64_pd(r0, _MM_SHUFFLE(3, 1, 2, 0));
        __m256d t2 = _mm256_permute4x64_pd(r1, _MM_SHUFFLE(3, 1, 2, 0));

        // Blend 128-bit lanes
        // A_hi = [h0, h1, h2, h3]
        __m256d a_hi = _mm256_permute2f128_pd(t1, t2, 0x20);
        // A_lo = [l0, l1, l2, l3]
        __m256d a_lo = _mm256_permute2f128_pd(t1, t2, 0x31);

        // Store to buffer (stream A_hi then A_lo)
        _mm256_storeu_pd(buffer, a_hi);
        _mm256_storeu_pd(buffer + 4, a_lo);
        buffer += 8;
    }
}

// Pack B: Extract KC x NR block.
// Since NR=1, we just pack scalars for broadcast.
// Stored as pairs [hi, lo], [hi, lo]... sequentially
void pack_B(int k, const float62x2 *B, int ldb, double *buffer) {
    for (int p = 0; p < k; ++p) {
        // Just copy the struct as is. 
        // B is accessed as scalars in the kernel and broadcasted.
        const double* ptr = (const double*)(B + p);
        buffer[0] = ptr[0]; // hi
        buffer[1] = ptr[1]; // lo
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
void micro_kernel(int k, const double *packA, const double *packB, 
                  float62x2 *C, int ldc) {
    
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
    double* c_ptr = (double*)C;
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
    // unpacklo(c_hi, c_lo) -> [h0, l0, h1, l1]
    __m256d res0 = _mm256_unpacklo_pd(c_hi, c_lo);
    // unpackhi(c_hi, c_lo) -> [h2, l2, h3, l3]
    __m256d res1 = _mm256_unpackhi_pd(c_hi, c_lo);

    _mm256_storeu_pd(c_ptr, res0);
    _mm256_storeu_pd(c_ptr + 4, res1);
}

// ============================================================================
// Macro-Kernel and Tiling
// ============================================================================

void gemm_dd_custom(int M, int N, int K, 
                    const float62x2 *A, int lda, 
                    const float62x2 *B, int ldb, 
                    float62x2 *C, int ldc) {
    
    // Parallelize Loop 5 (jc)
    #pragma omp parallel
    {
        // Thread-local packing buffers
        // Pack A needs: MC * KC * sizeof(float62x2)
        std::vector<double> packA_buf(MC * KC * 2); 
        // Pack B needs: KC * NC * sizeof(float62x2)
        // Since NR=1, we pack strips of B.
        // Actually, for Loop 4, we pack B once per KC block.
        // Let's use a simpler approach: Pack B inside the jc loop.
        std::vector<double> packB_buf(KC * NC * 2);

        #pragma omp for collapse(2)
        for (int jc = 0; jc < N; jc += NC) {
            for (int kc = 0; kc < K; kc += KC) {
                int nc = std::min(NC, N - jc);
                int k_block = std::min(KC, K - kc);
                
                // Pack B: Block B(kc:kc+k_block, jc:jc+nc)
                // We pack column by column for B? 
                // Our microkernel iterates K, so B should be packed such that
                // K dimension is contiguous for each column, or NR strip.
                // Since NR=1, we just pack columns.
                double *pB = packB_buf.data();
                for (int j = 0; j < nc; ++j) {
                    pack_B(k_block, &B[(jc + j)*ldb + kc], ldb, pB + j * k_block * 2);
                }

                for (int ic = 0; ic < M; ic += MC) {
                    int mc = std::min(MC, M - ic);

                    // Pack A: Block A(ic:ic+mc, kc:kc+k_block)
                    // We pack MR strips.
                    double *pA = packA_buf.data();
                    for (int i = 0; i < mc; i += MR) {
                         // Check boundary for MR
                         if (i + MR > mc) break; // Handle edge case (omitted for brevity)
                         pack_A(k_block, &A[ic + i + kc*lda], lda, pA + i * k_block * 2);
                    }

                    // Macro Kernel: Iterate over packed buffers
                    for (int jr = 0; jr < nc; jr += NR) { // NR=1
                        for (int ir = 0; ir < mc; ir += MR) { // MR=4
                             if (ir + MR > mc) continue; // Edge case
                             
                             micro_kernel(k_block, 
                                          packA_buf.data() + ir * k_block * 2, 
                                          packB_buf.data() + jr * k_block * 2, 
                                          &C[(jc + jr)*ldc + (ic + ir)], 
                                          ldc);
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Main / Test
// ============================================================================

int main() {
    int M = 1024;
    int N = 1024;
    int K = 2048;
    
    std::vector<float62x2> A(M * K);
    std::vector<float62x2> B(K * N);
    std::vector<float62x2> C(M * N);

    // Initialize (Column Major for standard BLAS compat)
    // A(i, k) -> A[k * M + i] if ColMajor.
    // Our code assumes generic strided access.
    // Let's assume Column Major: LDA = M, LDB = K, LDC = M
    
    // Init data...
    #pragma omp parallel for
    for(int i=0; i<M*K; ++i) { A[i].limbs[0] = 1.0; A[i].limbs[1] = 1e-18; }
    #pragma omp parallel for
    for(int i=0; i<K*N; ++i) { B[i].limbs[0] = 1.0; B[i].limbs[1] = 1e-18; }
    #pragma omp parallel for
    for(int i=0; i<M*N; ++i) { C[i].limbs[0] = 0.0; C[i].limbs[1] = 0.0; }

    std::cout << "Starting GEMM..." << std::endl;
    gemm_dd_custom(M, N, K, A.data(), M, B.data(), K, C.data(), M);
    std::cout << "Done." << std::endl;

    return 0;
}