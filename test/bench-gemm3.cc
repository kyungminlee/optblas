#include <vector>
#include "../src/opt/Float64x2.hh"


#define RESTRICT
using real = float64x2;

extern "C"
void ddgemm_(char const * ptransa, char const * ptransb,
    int const * pm, int const * pn, int const * pk,
    real const * palpha,
    const real * RESTRICT A, int const * plda,
    const real * RESTRICT B, int const * pldb,
    real const * pbeta,
    real * RESTRICT C, int const * pldc);


int main() {
    int M = 1024;
    int N = 1024;
    int K = 2048;
    
    std::vector<float64x2> A(M * K);
    std::vector<float64x2> B(K * N);
    std::vector<float64x2> C(M * N);

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

    float64x2 alpha{1.0, 0.0};
    float64x2 beta{0.0, 0.0};
    std::cout << "Starting GEMM..." << std::endl;
    ddgemm_("N", "N", &M, &N, &K, &alpha, A.data(), &M, B.data(), &K,
    &beta, C.data(), &M);
    std::cout << "Done." << std::endl;

    return 0;
}