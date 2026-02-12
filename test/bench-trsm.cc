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
    const char *lib_path2 = "src/opt/libblas.so"; // Your implementation

    void *handle2 = dlopen(lib_path2, RTLD_LAZY);
    if (!handle2) {
        std::cerr << "Cannot open library 2 (" << lib_path2 << "): " << dlerror() << std::endl;
        return 1;
    }

    dlerror(); // Clear any existing errors


    dtrsm_t dtrsm2 = (dtrsm_t)dlsym(handle2, "dtrsm_");
    const char *err2 = dlerror();
    if (err2) {
        std::cerr << "Cannot load symbol 'dtrsm_' from library 2: " << err2 << std::endl;
        return 1;
    }

    // Random setup
    std::mt19937 gen(0);
    std::uniform_int_distribution<> prob_dist(0, 10);
    std::uniform_int_distribution<> dim_dist(1024, 2048);
    std::uniform_int_distribution<> pad_dist(0, 64);
    std::uniform_int_distribution<> opt_dist(0, 1);
    std::uniform_int_distribution<> trans_dist(0, 2);
    std::normal_distribution<> alpha_dist;

    const char sides[] = {'L', 'R'};
    const char uplos[] = {'U', 'L'};
    const char trans[] = {'N', 'T', 'C'};
    const char diags[] = {'N', 'U'};

    int iterations = 10;
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
        std::vector<double> B2(ldb * n);

        fill_random(A, gen);
        fill_random(B2, gen);
        
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

        // Run Reference
        auto t2 = std::chrono::high_resolution_clock::now();

        // Run Implementation
	for (int r = 0; r < 10; ++r) {
		dtrsm2(&side, &uplo, &transa, &diag, &m, &n, &alpha, A.data(), &lda, B2.data(), &ldb);
	}
        auto t3 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> duration2 = t3 - t2;
	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(duration2).count() * 1e-6 << std::endl;
    }

    dlclose(handle2);
    return 0;
}
