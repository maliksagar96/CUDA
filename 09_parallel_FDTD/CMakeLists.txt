cmake_minimum_required(VERSION 3.10)
project(FDTD LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})
find_package(jsoncpp REQUIRED)

include_directories(/usr/include/opencascade include)

set(SOURCES
    src/main.cpp
    src/Mesh2D.cpp
    # src/vtkWriter.cpp
    # src/fdtd_2d.cpp
    src/fdtd_2d.cu
    src/global_variables.cpp
)

add_executable(fdtd ${SOURCES})

set(OCC_LIBS
    TKernel
    TKMath
    TKBRep
    TKSTEP
    TKSTEPAttr
    TKSTEP209
    TKSTEPBase
    TKIGES
    TKXSBase
    TKGeomBase
    TKGeomAlgo
    TKTopAlgo
)

set_target_properties(fdtd PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(fdtd PRIVATE include /usr/include/opencascade)
target_link_libraries(fdtd PRIVATE
    ${VTK_LIBRARIES}
    ${OCC_LIBS}
    jsoncpp_lib
)
