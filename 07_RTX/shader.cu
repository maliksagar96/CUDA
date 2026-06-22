#include <optix.h>
#include <optix_device.h>

struct Params
{
    unsigned int* result;
    OptixTraversableHandle handle;
};

extern "C" __constant__ Params params;

//----------------------------------------------------
// Ray Generation
//----------------------------------------------------
extern "C" __global__ void __raygen__rg()
{
    unsigned int hit = 0;

    float3 rayOrigin    = make_float3(0.0f, 0.0f, -1.0f);
    float3 rayDirection = make_float3(0.0f, 0.0f,  1.0f);

    optixTrace(
        params.handle,
        rayOrigin,
        rayDirection,
        0.0f,                  // tmin
        1e16f,                 // tmax
        0.0f,                  // rayTime
        255,                   // visibilityMask
        OPTIX_RAY_FLAG_NONE,
        0,                     // SBT offset
        1,                     // SBT stride
        0,                     // missSBTIndex
        hit                    // payload
    );

    params.result[0] = hit;
}

//----------------------------------------------------
// Closest Hit
//----------------------------------------------------
extern "C" __global__ void __closesthit__ch()
{
    optixSetPayload_0(1);
}

//----------------------------------------------------
// Miss
//----------------------------------------------------
extern "C" __global__ void __miss__ms()
{
    optixSetPayload_0(0);
}