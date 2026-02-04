#include <iostream>
#include <vector>
#include <random>
#include <dlfcn.h>
#include <cmath>
#include <chrono>

typedef void (*dgemm_t)(const char* transa, const char* transb,
                        const int* m, const int* n, const int* k,
                        const double* alpha, const double* a, const int* lda,
                        const double* b, const int* ldb,
                        const double* beta, double* c, const int* ldc);
extern "C" dgemm_t dgemm_;

void fill_random(std::vector<double>& vec, std::mt19937& gen) {
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (auto& val : vec) {
        val = dis(gen);
    }
}

int main() {
    const char *lib_path1 = "src/refblas/libblasf.dylib";   // Reference BLAS
    const char *lib_path2 = "src/opt/libblas.dylib"; // Your implementation

    void* handle1 = dlopen(lib_path1, RTLD_LAZY);
    if (!handle1) {
        std::cerr << "Cannot open library 1: " << dlerror() << std::endl;
        return 1;
    }

    void* handle2 = dlopen(lib_path2, RTLD_LAZY);
    if (!handle2) {
        std::cerr << "Cannot open library 2: " << dlerror() << std::endl;
        dlclose(handle1);
        return 1;
    }

    dlerror(); // Clear any existing errors

    dgemm_t dgemm1 = (dgemm_t)dlsym(handle1, "dgemm_");
    const char* err1 = dlerror();
    if (err1) {
        std::cerr << "Cannot load symbol 'dgemm_' from library 1: " << err1 << std::endl;
        return 1;
    }

    dgemm_t dgemm2 = (dgemm_t)dlsym(handle2, "dgemm_");
    // dgemm_t dgemm2 = dgemm_;
    const char* err2 = dlerror();
    if (err2) {
        std::cerr << "Cannot load symbol 'dgemm_' from library 2: " << err2 << std::endl;
        return 1;
    }

    // Stress test parameters
    const int iterations = 1000;
    // const double alpha = 1.0, beta = 0.0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> prob_dist(0, 2);
    std::uniform_int_distribution<> dim_dist(2, 512);
    std::uniform_int_distribution<> pad_dist(1, 128);
    std::uniform_int_distribution<> trans_dist(0, 5);
    const std::vector<char> trans_opts = {'N', 'T', 'C', 'n', 't', 'c'};

    std::vector<double> A, B, C1, C2;

    std::cout << "Starting stress test: " << iterations << " iterations with randomized sizes and transposes..." << std::endl;

    for (int i = 0; i < iterations; ++i) {
        int M = (prob_dist(gen) == 0) ? 1 : dim_dist(gen);
        int N = (prob_dist(gen) == 0) ? 1 : dim_dist(gen);
        int K = (prob_dist(gen) == 0) ? 1 : dim_dist(gen);

        char transA = trans_opts[trans_dist(gen)];
        char transB = trans_opts[trans_dist(gen)];

        bool trA = (transA == 'N' || transA == 'n') ? false : true;
        bool trB = (transB == 'N' || transB == 'n') ? false : true;

        int rowsA = trA ? K : M;
        int colsA = trA ? M : K;
        int rowsB = trB ? N : K;
        int colsB = trB ? K : N;

        int lda = (prob_dist(gen) == 0) ? rowsA : rowsA + pad_dist(gen);
        int ldb = (prob_dist(gen) == 0) ? rowsB : rowsB + pad_dist(gen);
        int ldc = (prob_dist(gen) == 0) ? M : M + pad_dist(gen);

        std::cout << "Iteration " << (i + 1) << ": M=" << M << ", N=" << N << ", K=" << K
                  << ", transA=" << transA << ", transB=" << transB
                  << ", lda=" << lda << ", ldb=" << ldb << ", ldc=" << ldc << std::endl;

        A.resize(lda * colsA);
        B.resize(ldb * colsB);
        C1.resize(ldc * N);
        C2.resize(ldc * N);

        fill_random(A, gen);
        fill_random(B, gen);
        fill_random(C1, gen);
        C2 = C1;
        std::normal_distribution<double> dis;

        double alpha = dis(gen);
        double beta = dis(gen);

        auto t1 = std::chrono::high_resolution_clock::now();
        dgemm1(&transA, &transB, &M, &N, &K, &alpha, A.data(), &lda, B.data(), &ldb, &beta, C1.data(), &ldc);
        auto t2 = std::chrono::high_resolution_clock::now();
        dgemm2(&transA, &transB, &M, &N, &K, &alpha, A.data(), &lda, B.data(), &ldb, &beta, C2.data(), &ldc);
        auto t3 = std::chrono::high_resolution_clock::now();
        
        // Compare results
        double max_diff = 0.0;
        for (int c = 0; c < N; ++c) {
            for (int r = 0; r < M; ++r) {
                double diff = std::abs(C1[r + c * ldc] - C2[r + c * ldc]);
                if (diff > max_diff) max_diff = diff;
            }
        }

        if (max_diff > 1e-10) {
            std::cerr << "Mismatch at iteration " << (i + 1) << ": max_diff = " << max_diff << std::endl;
            std::cerr << "Dimensions: M=" << M << ", N=" << N << ", K=" << K
                      << ", lda=" << lda << ", ldb=" << ldb << ", ldc=" << ldc << std::endl;
            return 1;
        }
        std::chrono::duration<double> duration1 = t2 - t1;
        std::chrono::duration<double> duration2 = t3 - t2;
        std::cout << std::scientific;
        std::cout << "Iteration " << (i + 1) << " passed. Time1: " << duration1.count()
                  << " s, Time2: " << duration2.count() << " s." << std::endl;
    }

    std::cout << "Stress test completed successfully." << std::endl;

    dlclose(handle1);
    dlclose(handle2);
    return 0;
}