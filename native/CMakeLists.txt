cmake_minimum_required(VERSION 3.18)
project(lowram_kernel LANGUAGES CXX)

add_library(lowram_kernel SHARED lowram_kernel.cpp)
target_compile_features(lowram_kernel PRIVATE cxx_std_20)
target_compile_options(lowram_kernel PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -ffast-math>)
set_target_properties(lowram_kernel PROPERTIES OUTPUT_NAME lowram_kernel)
