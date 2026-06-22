#include <iostream>
#include <fstream>
#include <vector>

#include <cuda_runtime.h>

#include <optix.h>
#include <optix_stubs.h>

#define CUDA_CHECK(call)                                         \
do {                                                             \
    cudaError_t err = call;                                      \
    if(err != cudaSuccess) {                                     \
        std::cerr << "CUDA Error: "                              \
                  << cudaGetErrorString(err) << std::endl;       \
        exit(EXIT_FAILURE);                                      \
    }                                                            \
} while(0)

#define OPTIX_CHECK(call)                                        \
do {                                                             \
    OptixResult res = call;                                      \
    if(res != OPTIX_SUCCESS) {                                   \
        std::cerr << "OptiX Error: "                             \
                  << res << std::endl;                           \
        exit(EXIT_FAILURE);                                      \
    }                                                            \
} while(0)

int main() {
    CUDA_CHECK(cudaFree(0));
    OPTIX_CHECK(optixInit());
    CUcontext cuCtx = 0;
    OptixDeviceContext context;
    OptixDeviceContextOptions options = {};
    OPTIX_CHECK( optixDeviceContextCreate(cuCtx, &options, &context));

    std::cout << "OptiX Context Created\n";

    optixDeviceContextDestroy(context);

    return 0;
}