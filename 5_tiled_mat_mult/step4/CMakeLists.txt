cmake_minimum_required(VERSION 3.18)

project(cublas_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Optional but recommended
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Find CUDA and cuBLAS
find_package(CUDAToolkit REQUIRED)

add_executable(cublas_benchmark
  04_cublas_benchmark.cu
)

# Link cuBLAS and CUDA runtime
target_link_libraries(cublas_benchmark
  CUDA::cudart
  CUDA::cublas
)

# Optimization flags
target_compile_options(cublas_benchmark PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# GPU architecture (important!)
# For GTX 1650 -> sm_75
set_target_properties(cublas_benchmark PROPERTIES
  CUDA_ARCHITECTURES 75
)
