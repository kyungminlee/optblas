/*
 * test-trsm.c
 * Test harness for DTRSM implementation
 * Generated based on test.cc pattern
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <dlfcn.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>

// Typedef for dtrsm function pointer (Fortran convention: pass by reference)
typedef void (*dtrsm_t)(const char *side, const char *uplo, const char *transa, const char *diag,
                        const int *m, const int *n, const double *alpha,
                        const double *A, const int *lda, double *B, const int *ldb);

// Helper to get current time in seconds
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Helper to fill matrix with random values
void fill_random(double *A, int size) {
    for (int i = 0; i < size; ++i) {
        A[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    }
}

int main(int argc, char **argv) {
    const char *lib_path1 = "./libblas.dylib";   // Reference BLAS
    const char *lib_path2 = "./libblasf.dylib"; // Your implementation

    // Load libraries
    void *handle1 = dlopen(lib_path1, RTLD_LAZY);
    if (!handle1) {
        fprintf(stderr, "Cannot open library 1 (%s): %s\n", lib_path1, dlerror());
        return 1;
    }

    void *handle2 = dlopen(lib_path2, RTLD_LAZY);
    if (!handle2) {
        fprintf(stderr, "Cannot open library 2 (%s): %s\n", lib_path2, dlerror());
        dlclose(handle1);
        return 1;
    }

    dlerror(); // Clear any existing errors

    // Load symbols
    dtrsm_t dtrsm1 = (dtrsm_t)dlsym(handle1, "dtrsm_");
    const char *err1 = dlerror();
    if (err1) {
        fprintf(stderr, "Cannot load symbol 'dtrsm_' from library 1: %s\n", err1);
        return 1;
    }

    dtrsm_t dtrsm2 = (dtrsm_t)dlsym(handle2, "dtrsm_");
    const char *err2 = dlerror();
    if (err2) {
        fprintf(stderr, "Cannot load symbol 'dtrsm_' from library 2: %s\n", err2);
        return 1;
    }

    srand(time(NULL));

    int iterations = 100;
    printf("Starting stress test: %d iterations with randomized sizes and parameters...\n", iterations);

    const char sides[] = {'L', 'R'};
    const char uplos[] = {'U', 'L'};
    const char trans[] = {'N', 'T', 'C'};
    const char diags[] = {'N', 'U'};

    for (int i = 0; i < iterations; ++i) {
        // Randomize dimensions
        int m = rand() % 512 + 1;
        int n = rand() % 512 + 1;
        
        // Randomize options
        char side = sides[rand() % 2];
        char uplo = uplos[rand() % 2];
        char transa = trans[rand() % 3];
        char diag = diags[rand() % 2];

        // Determine K (size of A)
        int k = (side == 'L' || side == 'l') ? m : n;
        
        // Randomize strides (must be >= dimension)
        int lda = k + rand() % 64;
        int ldb = m + rand() % 64;

        double alpha = (double)rand() / RAND_MAX;

        printf("Iteration %d: M=%d, N=%d, Side=%c, Uplo=%c, TransA=%c, Diag=%c, lda=%d, ldb=%d\n",
               i + 1, m, n, side, uplo, transa, diag, lda, ldb);

        // Allocate memory
        double *A = (double *)malloc(lda * k * sizeof(double));
        double *B1 = (double *)malloc(ldb * n * sizeof(double));
        double *B2 = (double *)malloc(ldb * n * sizeof(double));

        if (!A || !B1 || !B2) {
            fprintf(stderr, "Memory allocation failed\n");
            return 1;
        }

        fill_random(A, lda * k);
        fill_random(B1, ldb * n);
        
        // Ensure A is diagonally dominant to avoid singularity/instability
        // especially for non-unit diagonal cases.
        if (diag == 'N' || diag == 'n') {
             for (int j = 0; j < k; ++j) {
                 // Add k to diagonal to ensure dominance
                 A[j + j * lda] += (k + 2.0) * ((A[j + j * lda] > 0) ? 1.0 : -1.0);
             }
        }

        // Copy B1 to B2 for the second run
        memcpy(B2, B1, ldb * n * sizeof(double));

        // Run Reference
        double t1 = get_time();
        dtrsm1(&side, &uplo, &transa, &diag, &m, &n, &alpha, A, &lda, B1, &ldb);
        double t2 = get_time();

        // Run Implementation
        dtrsm2(&side, &uplo, &transa, &diag, &m, &n, &alpha, A, &lda, B2, &ldb);
        double t3 = get_time();

        // Compare results
        double max_diff = 0.0;
        for (int c = 0; c < n; ++c) {
            for (int r = 0; r < m; ++r) {
                double diff = fabs(B1[r + c * ldb] - B2[r + c * ldb]);
                if (diff > max_diff) max_diff = diff;
            }
        }

        printf("  Time1: %.6f s, Time2: %.6f s, Max Diff: %.6e\n", t2 - t1, t3 - t2, max_diff);

        // Check for mismatch
        if (max_diff > 1e-6) {
            fprintf(stderr, "Mismatch at iteration %d: max_diff = %.6e\n", i + 1, max_diff);
            fprintf(stderr, "Params: M=%d, N=%d, Side=%c, Uplo=%c, TransA=%c, Diag=%c\n", 
                    m, n, side, uplo, transa, diag);
            free(A); free(B1); free(B2);
            return 1;
        }

        free(A);
        free(B1);
        free(B2);
    }

    printf("Stress test completed successfully.\n");
    dlclose(handle1);
    dlclose(handle2);
    return 0;
}