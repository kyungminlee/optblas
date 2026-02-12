#include "Float64x2.hh"
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstdlib>

#ifndef RESTRICT
#define RESTRICT
#endif

extern "C"
void ddgemm_(char const * ptransa, char const * ptransb,
    int const * pm, int const * pn, int const * pk,
    float64x2 const * palpha,
    const float64x2 * RESTRICT A, int const * plda,
    const float64x2 * RESTRICT B, int const * pldb,
    float64x2 const * pbeta,
    float64x2 * RESTRICT C, int const * pldc);


void ref_gemm(const char *transA, const char *transB, 
              int M, int N, int K,
              float64x2 alpha, const float64x2 *A, int lda,
              const float64x2 *B, int ldb,
              float64x2 beta, float64x2 *C, int ldc) {
    bool tA = (*transA == 'T' || *transA == 't' || *transA == 'C' || *transA == 'c');
    bool tB = (*transB == 'T' || *transB == 't' || *transB == 'C' || *transB == 'c');

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            float64x2 sum(0.0, 0.0);
            for (int l = 0; l < K; ++l) {
                float64x2 a_val = tA ? A[l + i * lda] : A[i + l * lda];
                float64x2 b_val = tB ? B[j + l * ldb] : B[l + j * ldb];
                sum = sum + a_val * b_val;
            }
            if (beta.limbs[0] == 0.0 && beta.limbs[1] == 0.0) {
                C[i + j * ldc] = alpha * sum;
            } else {
                C[i + j * ldc] = alpha * sum + beta * C[i + j * ldc];
            }
        }
    }
}

void test_gemm(const char *transA, const char *transB, 
               const int *m, const int *n, const int *k,
               const float64x2 *alpha, const float64x2 *A, const int *lda,
               const float64x2 *B, const int *ldb,
               const float64x2 *beta, float64x2 *C, const int *ldc) {
    
    int M = *m, N = *n, K = *k;
    int LDA = *lda, LDB = *ldb, LDC = *ldc;

    // Make a copy of C for reference
    std::vector<float64x2> C_ref(LDC * N);
    for(int j=0; j<N; ++j) {
        for(int i=0; i<M; ++i) {
            C_ref[i + j*LDC] = C[i + j*LDC];
        }
    }

    // Run Optimized GEMM
    ddgemm_(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

    // Run Reference GEMM
    ref_gemm(transA, transB, M, N, K, *alpha, A, LDA, B, LDB, *beta, C_ref.data(), LDC);

    // Compare
    double max_err = 0.0;
    for(int j=0; j<N; ++j) {
        for(int i=0; i<M; ++i) {
            float64x2 diff = C[i + j*LDC] - C_ref[i + j*LDC];
            max_err = std::max(max_err, std::abs(diff.limbs[0]));
        }
    }

    std::cout << "Test Result: ";
    if (max_err < 1e-12) {
        std::cout << "PASS";
    } else {
        std::cout << "FAIL";
    }
    std::cout << " (Max Error: " << max_err << ")" << std::endl;
}

void test_full() {
    const char* trans[] = {"N", "T"};
    // Shapes: M, N, K
    // one, two, very large
    int shapes[][3] = {
        {1, 1, 1},
        {2, 2, 2},
        {17, 17, 17},
        {128, 128, 128},
        {256, 256, 256}
    };
    
    float64x2 alpha = {1.0, 0.0};
    float64x2 beta = {0.0, 0.0};

    for (const char* ta : trans) {
        for (const char* tb : trans) {
            for (auto& s : shapes) {
                int M = s[0];
                int N = s[1];
                int K = s[2];

                // Strides: contiguous (0) and discontiguous (5)
                int offsets[] = {0, 5};
                
                for (int offA : offsets) {
                    for (int offB : offsets) {
                        int min_lda = (*ta == 'N') ? M : K;
                        int min_ldb = (*tb == 'N') ? K : N;
                        int LDA = min_lda + offA;
                        int LDB = min_ldb + offB;
                        int LDC = M; 

                        int colsA = (*ta == 'N') ? K : M;
                        int colsB = (*tb == 'N') ? N : K;
                        
                        std::vector<float64x2> A(LDA * colsA);
                        std::vector<float64x2> B(LDB * colsB);
                        std::vector<float64x2> C(LDC * N);

                        for(auto& v : A) { v.limbs[0] = ((double)rand()/RAND_MAX); v.limbs[1] = 0; }
                        for(auto& v : B) { v.limbs[0] = ((double)rand()/RAND_MAX); v.limbs[1] = 0; }
                        for(auto& v : C) { v.limbs[0] = 0; v.limbs[1] = 0; }

                        std::cout << "M=" << M << " N=" << N << " K=" << K 
                                  << " TA=" << ta << " TB=" << tb 
                                  << " LDA=" << LDA << " LDB=" << LDB << " ";
                        
                        test_gemm(ta, tb, &M, &N, &K, &alpha, A.data(), &LDA, B.data(), &LDB, &beta, C.data(), &LDC);
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
    test_full();

    return 0;
}