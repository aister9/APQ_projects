#include <optix.h>

#include "optixRaycasting.h"
#include "optixRaycastingKernels.h"

#include "cuda/LocalGeometry.h"
#include "cuda/whitted.h"

#include <sutil/vec_math.h>


extern "C" {
    __constant__ Params params;
}


extern "C" __global__ void __raygen__from_buffer()
{
    const uint3        idx = optixGetLaunchIndex();
    const uint3        dim = optixGetLaunchDimensions();
    const unsigned int linear_idx = idx.z * dim.y * dim.x + idx.y * dim.x + idx.x;

    unsigned int t, nx, ny, nz;
    Ray          ray = params.rays[linear_idx];
    optixTrace(params.handle, ray.origin, ray.dir, ray.tmin, ray.tmax, 0.0f, OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE, RAY_TYPE_RADIANCE, RAY_TYPE_COUNT, RAY_TYPE_RADIANCE, t, nx, ny, nz);

    Hit hit;
    hit.t = int_as_float(t);
    hit.geom_normal.x = int_as_float(nx);
    hit.geom_normal.y = int_as_float(ny);
    hit.geom_normal.z = int_as_float(nz);
    params.hits[linear_idx] = hit;
}


extern "C" __global__ void __miss__buffer_miss()
{
    optixSetPayload_0(float_as_int(-1.0f));
    optixSetPayload_1(float_as_int(1.0f));
    optixSetPayload_2(float_as_int(0.0f));
    optixSetPayload_3(float_as_int(0.0f));
}


extern "C" __global__ void __closesthit__buffer_hit()
{
    const unsigned int t = optixGetRayTmax();

    whitted::HitGroupData* rt_data = (whitted::HitGroupData*)optixGetSbtDataPointer();
    LocalGeometry          geom = getLocalGeometry(rt_data->geometry_data);

    // Set the hit data
    optixSetPayload_0(float_as_int(t));
    optixSetPayload_1(float_as_int(geom.N.x));
    optixSetPayload_2(float_as_int(geom.N.y));
    optixSetPayload_3(float_as_int(geom.N.z));
}


extern "C" __global__ void __anyhit__texture_mask()
{
    whitted::HitGroupData* rt_data = (whitted::HitGroupData*)optixGetSbtDataPointer();
    LocalGeometry          geom = getLocalGeometry(rt_data->geometry_data);

    float4 mask = tex2D<float4>(rt_data->material_data.pbr.base_color_tex, geom.UV.x, geom.UV.y);
    if (mask.x < 0.5f && mask.y < 0.5f)
    {
        optixIgnoreIntersection();
    }
}

