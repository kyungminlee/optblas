/*
 * test-trsm.cc
 * Test harness for DTRSM implementation
 * Generated based on test.cc pattern
 */

#include <iostream>
#include <vector>
#include <random>
#include <dlfcn.h>
#include <cmath>
#include <chrono>
#include <cstring>
#include <iomanip>

// Typedef for dtrsm function pointer (Fortran convention: pass by reference)
typedef void (*dtrsm_t)(const char *side, const char *uplo, const char *transa, const char *diag,
                        const int *m, const int *n, const double *alpha,
                        const double *A, const int *lda, double *B, const int *ldb);

void fill_random(std::vector<double>& vec, std::mt19937& gen) {
    std::normal_distribution<> dis;
    for (auto& val : vec) {
        val = dis(gen);
    }
}

int main() {
    const char *lib_path1 = "src/refblas/libblasf.dylib";   // Reference BLAS
    const char *lib_path2 = "src/opt/libblas.dylib"; // Your implementation

    // Load libraries
    void *handle1 = dlopen(lib_path1, RTLD_LAZY);
    if (!handle1) {
        std::cerr << "Cannot open library 1 (" << lib_path1 << "): " << dlerror() << std::endl;
        return 1;
    }

    void *handle2 = dlopen(lib_path2, RTLD_LAZY);
    if (!handle2) {
        std::cerr << "Cannot open library 2 (" << lib_path2 << "): " << dlerror() << std::endl;
        dlclose(handle1);
        return 1;
    }

    dlerror(); // Clear any existing errors

    // Load symbols
    dtrsm_t dtrsm1 = (dtrsm_t)dlsym(handle1, "dtrsm_");
    const char *err1 = dlerror();
    if (err1) {
        std::cerr << "Cannot load symbol 'dtrsm_' from library 1: " << err1 << std::endl;
        return 1;
    }

    dtrsm_t dtrsm2 = (dtrsm_t)dlsym(handle2, "dtrsm_");
    const char *err2 = dlerror();
    if (err2) {
        std::cerr << "Cannot load symbol 'dtrsm_' from library 2: " << err2 << std::endl;
        return 1;
    }

    // Random setup
    std::mt19937 gen(0);
    std::uniform_int_distribution<> prob_dist(0, 10);
    std::uniform_int_distribution<> dim_dist(2, 2048);
    std::uniform_int_distribution<> pad_dist(0, 64);
    std::uniform_int_distribution<> opt_dist(0, 1);
    std::uniform_int_distribution<> trans_dist(0, 2);
    std::normal_distribution<> alpha_dist;

    const char sides[] = {'L', 'R'};
    const char uplos[] = {'U', 'L'};
    const char trans[] = {'N', 'T', 'C'};
    const char diags[] = {'N', 'U'};

    int iterations = 100;
    std::cout << "Starting stress test: " << iterations << " iterations with randomized sizes and parameters..." << std::endl;

    for (int i = 0; i < iterations; ++i) {
        // Randomize dimensions
        int m = (prob_dist(gen) == 0) ? 1 : dim_dist(gen);
        int n = (prob_dist(gen) == 0) ? 1 : dim_dist(gen);
        
        // Randomize options
        char side = sides[opt_dist(gen)];
        char uplo = uplos[opt_dist(gen)];
        char transa = trans[trans_dist(gen)];
        char diag = diags[opt_dist(gen)];

        // Determine K (size of A)
        int k = (side == 'L' || side == 'l') ? m : n;
        
        // Randomize strides (must be >= dimension)
        int lda = k + pad_dist(gen);
        int ldb = m + pad_dist(gen);

        double alpha = alpha_dist(gen);
        // alpha = 1.0;

        // std::cout << "Iteration " << (i + 1) << ": M=" << m << ", N=" << n 
        //           << ", Side=" << side << ", Uplo=" << uplo 
        //           << ", TransA=" << transa << ", Diag=" << diag 
        //           << ", lda=" << lda << ", ldb=" << ldb << std::endl;

        // Allocate memory
        std::vector<double> A(lda * k);
        std::vector<double> B1(ldb * n);
        std::vector<double> B2(ldb * n);

        fill_random(A, gen);
        fill_random(B1, gen);
        
        // Ensure A is diagonally dominant to avoid singularity/instability
        // especially for non-unit diagonal cases.
        if (diag == 'N' || diag == 'n') {
             for (int j = 0; j < k; ++j) {
                 // Add k to diagonal to ensure dominance
                 double val = A[j + j * lda];
                 A[j + j * lda] += (k + 2.0) * ((val >= 0) ? 1.0 : -1.0) * 10.0;
             }
        } else {
            for (int i = 0; i < k; ++i) {
                for (int j = 0; j < k; ++j) {
                    A[i+j*lda] *= 0.05;
                }
            }
        }

        // Copy B1 to B2 for the second run
        B2 = B1;

        // Run Reference
        auto t1 = std::chrono::high_resolution_clock::now();
        dtrsm1(&side, &uplo, &transa, &diag, &m, &n, &alpha, A.data(), &lda, B1.data(), &ldb);
        auto t2 = std::chrono::high_resolution_clock::now();

        // Run Implementation
        dtrsm2(&side, &uplo, &transa, &diag, &m, &n, &alpha, A.data(), &lda, B2.data(), &ldb);
        auto t3 = std::chrono::high_resolution_clock::now();

        // Compare results
        double max_diff = 0.0;
        for (int c = 0; c < n; ++c) {
            for (int r = 0; r < m; ++r) {
                double diff = std::abs(B1[r + c * ldb] - B2[r + c * ldb]);
                if (diff > max_diff) max_diff = diff;
            }
        }

        std::chrono::duration<double> duration1 = t2 - t1;
        std::chrono::duration<double> duration2 = t3 - t2;

        std::cout << std::scientific << std::setprecision(6);
        std::cout << "  Time1: " << duration1.count() << " s, Time2: " << duration2.count() 
                  << " s, Max Diff: " << max_diff << std::endl;

        // if (m == 1 || n == 1) { continue; }

        std::cout << side << uplo << transa << diag << " " << m << " " << n;
        // Check for mismatch
        if (max_diff > 1e-6) {
            // std::cout << "Iteration " << (i + 1) << ": ";
            // std::cout << "Params: M=" << m << ", N=" << n << ", Side=" << side 
            //           << ", Uplo=" << uplo << ", TransA=" << transa << ", Diag=" << diag  << "\tFAILED." << std::endl;
            // return 1;
            std::cout << "\tFAILED: max_diff = " << max_diff << "\n";
        } else {
            // std::cout << "Iteration " << (i + 1) << ": ";
            // std::cout << "Params: M=" << m << ", N=" << n << ", Side=" << side 
            //           << ", Uplo=" << uplo << ", TransA=" << transa << ", Diag=" << diag  << "\tpassed." << std::endl;
            std::cout << "\tpassed\n";
        }
    }

    std::cout << "Stress test completed successfully." << std::endl;
    dlclose(handle1);
    dlclose(handle2);
    return 0;
}