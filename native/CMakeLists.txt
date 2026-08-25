cmake_minimum_required(VERSION 3.16)
project(lowram_kernel LANGUAGES CXX)

add_library(lowram_kernel SHARED lowram_kernel.cpp)
target_compile_features(lowram_kernel PRIVATE cxx_std_20)
target_compile_options(lowram_kernel PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -ffast-math -fno-exceptions -fno-rtti>)

# OpenMP is optional: the scalar single-threaded path remains portable on
# systems where a low-RAM deployment does not ship an OpenMP runtime.
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(lowram_kernel PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(lowram_kernel PRIVATE LOWRAM_HAS_OPENMP=1)
endif()

# Do not force -march=native: the same binary must remain safe on older CPUs.
# AVX2 kernels are selected at compile time when enabled by the toolchain, and
# the scalar path is retained for portable builds.
set_target_properties(lowram_kernel PROPERTIES OUTPUT_NAME lowram_kernel)
if(MSVC)
    target_compile_options(lowram_kernel PRIVATE /O2)
endif()
if(UNIX AND NOT APPLE)
    target_link_libraries(lowram_kernel PRIVATE m)
endif()
