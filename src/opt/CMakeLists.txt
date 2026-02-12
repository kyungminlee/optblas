add_library(blas SHARED
  # dgemm.c egemm.c qgemm.c
  # dtrsm.c etrsm.c qtrsm.c
  d.c e.c q.c xerbla.c
  dd.cc
)
# target_link_libraries(blas PRIVATE OpenMP::OpenMP_C)
target_compile_options(blas PRIVATE -fopenmp)
# target_compile_options(gemm PRIVATE -march=native -funroll-loops -Xpreprocessor -fopenmp)
# target_link_libraries(gemm PRIVATE omp)
# target_link_directories(gemm PRIVATE /opt/homebrew/opt/libomp/lib)
