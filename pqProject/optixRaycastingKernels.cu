
#include <cuda_runtime.h>

#include "optixRaycastingKernels.h"

#include <sutil/vec_math.h>


inline int idivCeil(int x, int y)
{
    return (x + y - 1) / y;
}


__global__ void createRaysOrthoKernel(Ray* rays, int width, int height, float x0, float y0, float z, float dx, float dy)
{
    const int rayx = threadIdx.x + blockIdx.x * blockDim.x;
    const int rayy = threadIdx.y + blockIdx.y * blockDim.y;
    if (rayx >= width || rayy >= height)
        return;

    const int idx = rayx + rayy * width;
    rays[idx].origin = make_float3(x0 + rayx * dx, y0 + rayy * dy, z);
    rays[idx].tmin = 0.0f;
    rays[idx].dir = make_float3(0, 0, 1);
    rays[idx].tmax = 1e34f;
}


// Note: uses left handed coordinate system
void createRaysOrthoOnDevice(Ray* rays_device, int width, int height, float3 bbmin, float3 bbmax, float padding)
{
    const float3 bbspan = bbmax - bbmin;
    float        dx = bbspan.x * (1 + 2 * padding) / width;
    float        dy = bbspan.y * (1 + 2 * padding) / height;
    float        x0 = bbmin.x - bbspan.x * padding + dx / 2;
    float        y0 = bbmin.y - bbspan.y * padding + dy / 2;
    float        z = bbmin.z - fmaxf(bbspan.z, 1.0f) * .001f;

    dim3 blockSize(32, 16);
    dim3 gridSize(idivCeil(width, blockSize.x), idivCeil(height, blockSize.y));
    createRaysOrthoKernel << <gridSize, blockSize >> > (rays_device, width, height, x0, y0, z, dx, dy);
}


__global__ void translateRaysKernel(Ray* rays, int count, float3 offset)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= count)
        return;

    rays[idx].origin = rays[idx].origin + offset;
}


void translateRaysOnDevice(Ray* rays_device, int count, float3 offset)
{
    const int blockSize = 512;
    const int blockCount = idivCeil(count, blockSize);
    translateRaysKernel << <blockCount, blockSize >> > (rays_device, count, offset);
}


__global__ void shadeHitsKernel(float3* image, int count, const Hit* hits)
{

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= count)
        return;

    const float3 backgroundColor = make_float3(0.2f, 0.2f, 0.2f);
    if (hits[idx].t < 0.0f)
    {
        image[idx] = backgroundColor;
    }
    else
    {
        image[idx] = 0.5f * hits[idx].geom_normal + make_float3(0.5f, 0.5f, 0.5f);
    }
}


void shadeHitsOnDevice(float3* image_device, int count, const Hit* hits_device)
{
    const int blockSize = 512;
    const int blockCount = idivCeil(count, blockSize);
    shadeHitsKernel << <blockCount, blockSize >> > (image_device, count, hits_device);
}

