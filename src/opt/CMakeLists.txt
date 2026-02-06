add_library(blas SHARED dgemm.c dtrsm.c

egemm.c
qgemm.c
)
target_link_libraries(blas PRIVATE OpenMP::OpenMP_C)
# target_compile_options(gemm PRIVATE -march=native -funroll-loops -Xpreprocessor -fopenmp)
# target_link_libraries(gemm PRIVATE omp)
# target_link_directories(gemm PRIVATE /opt/homebrew/opt/libomp/lib)
