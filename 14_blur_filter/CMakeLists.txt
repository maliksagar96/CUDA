cmake_minimum_required(VERSION 3.10)

project(opencvCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)

find_package(OpenCV REQUIRED)

add_executable(main main.cu)

target_link_libraries(main ${OpenCV_LIBS})




