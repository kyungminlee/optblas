cmake_minimum_required(VERSION 3.27)
project(gemm LANGUAGES C CXX Fortran)

find_package(OpenMP REQUIRED)

add_subdirectory(src/opt)

add_subdirectory(src/refblas)

# target_link_libraries(gemm PRIVATE OpenMP::OpenMP_C)
# target_compile_options(gemm PRIVATE ${OpenMP_C_FLAGS})
add_executable(test test/test.cc)
target_link_libraries(test PRIVATE dl)

add_executable(test-trsm test/test-trsm.cc)
target_link_libraries(test-trsm PRIVATE dl)


# message(STATUS "${CMAKE_Fortran_LINK_EXECUTABLE}")
# message(STATUS "${CMAKE_Fortran_COMPILER_LINKER}")
