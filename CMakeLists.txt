cmake_minimum_required(VERSION 3.27)
project(gemm LANGUAGES C CXX Fortran)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP REQUIRED)

add_subdirectory(src/opt)

add_subdirectory(src/refblas)

# target_link_libraries(gemm PRIVATE OpenMP::OpenMP_C)
# target_compile_options(gemm PRIVATE ${OpenMP_C_FLAGS})
add_executable(test test/test.cc)
target_link_libraries(test PRIVATE OpenMP::OpenMP_CXX dl)

add_executable(test-trsm test/test-trsm.cc)
target_link_libraries(test-trsm PRIVATE OpenMP::OpenMP_CXX dl)

add_executable(bench-trsm test/bench-trsm.cc)
target_link_libraries(bench-trsm PRIVATE OpenMP::OpenMP_CXX dl)

# message(STATUS "${CMAKE_Fortran_LINK_EXECUTABLE}")
# message(STATUS "${CMAKE_Fortran_COMPILER_LINKER}")


# add_executable(bench-gemm2 src/opt2/gemm2.cc)
# target_compile_options(bench-gemm2 PRIVATE -fopenmp -march=haswell -fno-math-errno)
# target_link_libraries(bench-gemm2 PRIVATE /opt/intel/oneapi/compiler/latest/lib/libiomp5.so)


add_library(ddgemmavx2 src/opt2/gemm2.cc)
target_include_directories(ddgemmavx2 PUBLIC src/opt)
target_compile_options(ddgemmavx2 PRIVATE ${OpenMP_CXX_FLAGS} -march=haswell -fno-math-errno)
# target_link_libraries(bench-gemm2 /opt/intel/oneapi/compiler/latest/lib/libiomp5.so)

add_executable(test-gemm test/test-gemm.cc)
target_link_libraries(test-gemm PRIVATE ddgemmavx2)
target_link_libraries(test-gemm PRIVATE /opt/intel/oneapi/compiler/latest/lib/libiomp5.so)

add_executable(bench-gemm3 test/bench-gemm3.cc)
target_compile_options(bench-gemm3 PRIVATE -fopenmp -march=haswell -fno-math-errno)
target_link_libraries(bench-gemm3 PRIVATE blas)
target_link_libraries(bench-gemm3 PRIVATE /opt/intel/oneapi/compiler/latest/lib/libiomp5.so)

