cmake_minimum_required(VERSION 3.20)

project(rt_triangle LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

set(OPTIX_ROOT "/workspace/NVIDIA-OptiX-SDK-9.1.0-linux64-x86_64")

include_directories(
    ${OPTIX_ROOT}/include
)

#----------------------------------------------------
# Compile shaders.cu -> shaders.ptx
#----------------------------------------------------

set(PTX_FILE ${CMAKE_BINARY_DIR}/shaders.ptx)

add_custom_command(
    OUTPUT ${PTX_FILE}

    COMMAND
    ${CUDAToolkit_NVCC_EXECUTABLE}
    --ptx
    ${CMAKE_SOURCE_DIR}/shaders.cu
    -o ${PTX_FILE}
    -I${OPTIX_ROOT}/include

    DEPENDS
    ${CMAKE_SOURCE_DIR}/shaders.cu

    COMMENT "Generating PTX"
)

add_custom_target(
    shaders_ptx ALL
    DEPENDS ${PTX_FILE}
)

#----------------------------------------------------
# Main executable
#----------------------------------------------------

add_executable(rt_triangle
    main.cpp
)

add_dependencies(rt_triangle shaders_ptx)

target_include_directories(rt_triangle PRIVATE
    ${OPTIX_ROOT}/include
)

target_link_libraries(rt_triangle
    CUDA::cudart
    CUDA::cuda_driver
)

#----------------------------------------------------
# Copy PTX beside executable
#----------------------------------------------------

add_custom_command(
    TARGET rt_triangle POST_BUILD

    COMMAND
    ${CMAKE_COMMAND} -E copy_if_different
    ${PTX_FILE}
    $<TARGET_FILE_DIR:rt_triangle>/shaders.ptx
)