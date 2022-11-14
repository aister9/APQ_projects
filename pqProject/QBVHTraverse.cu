#include "QBVH4.h"
#include "Ray.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "helperMath.h"

#include <iostream>


struct GPURay {
	float3 origin;
	float3 dir;
	float tmin;
	float tmax;
	bool isHit;
};

struct GPUQBVH4 {
	float3 start; // 12 byte
	float3 extent; // 12 byte
	AISTER_GRAPHICS_ENGINE::Box3ui8 boxData[4]; // 24 byte
	UINT8 childFlag; // 0000 0000 : isLeaf / isNull // 1byte

	union child { // 4byte
		UINT32 boxIdx; //offset next node index
		UINT32 triIdx; //offset triangle index
	};
	child childs[4]; // 16 byte
};

__device__ bool intersect(GPURay* ray, GPUQBVH4 node, int i) {
	float3 bMin;
	bMin.x = node.start.x + node.extent.x * node.boxData[i].qlower[0];
	bMin.y = node.start.y + node.extent.y * node.boxData[i].qlower[1];
	bMin.z = node.start.z + node.extent.z * node.boxData[i].qlower[2];
	float3 bMax;
	bMax.x = node.start.x + node.extent.x * node.boxData[i].qupper[0];
	bMax.y = node.start.y + node.extent.y * node.boxData[i].qupper[1];
	bMax.z = node.start.z + node.extent.z * node.boxData[i].qupper[2];

	float3 invDir = make_float3(1 / ray->dir.x, 1 / ray->dir.y, 1 / ray->dir.z);

	float3 tMin = (bMin-ray->origin) * invDir;
	float3 tMax = (bMax-ray->origin) * invDir;

	float3 tmp1 = fminf(tMin, tMax);
	float3 tmp2 = fmaxf(tMin, tMax);

	tMin = tmp1; tMax = tmp2;

	float _min = fmaxf(tMin.x, tMin.y);
	float _max = fminf(tMax.x, tMax.y);

	if (tMin.x > tMax.y || tMin.y > tMax.x) return false;
	if (_min > tMax.z || tMin.z > _max) return false;

	_min = fmaxf(_min, tMin.z);
	_max = fminf(_max, tMax.z);

	float3 minPts = ray->origin + _min * ray->dir;
	float3 maxPts = ray->origin + _max * ray->dir;

	if (fabsf(_min) > fabsf(_max))
	{
		_min += _max;
		_max = _min - _max;
		_min = _min - _max;
	}

	if (ray->tmin < _min)
		return false;

	return true;
}

__device__ bool intersect(GPURay* ray, float3 p0, float3 p1, float3 p2) {
	float3 e1 = p1 - p0;
	float3 e2 = p2 - p0;

	float3 h = cross(ray->dir, e2);
	float a = dot(e1, h);

	float epsillon = 0.0000001;

	if (a > -epsillon && a < epsillon) return false;

	float f = 1.f / a;
	float3 s = ray->origin - p0;
	float u = f * dot(s, h);

	if (u < 0.0 || u>1.0) return false;

	float3 q = cross(s, e1);
	float v = f * dot(ray->dir, q);

	if (v < 0.0 || u + v>1.0) return false;

	float t = f * dot(e2, q);

	if (t > epsillon) {
		if (ray->tmin > t) {
			ray->tmin = t;
			return true;
		}
		else return false;
	}
	else return false;
}

__global__ void traverse(GPURay* rays, GPUQBVH4* bvhs, float3 *vertices, int3 *trilist, int vsize, int tsize, int width, int height) {
	const int idxx = threadIdx.x + blockIdx.x * blockDim.x;
	const int idxy = threadIdx.y + blockIdx.y * blockDim.y;

	if (idxx >= width || idxy >= height)
		return;

	const int idx = idxx + idxy * width;

	int stack[64];
	int nodeind = 0;

	stack[0] = 0;

	while (nodeind != -1) {
		int dataind = stack[nodeind];
		nodeind--;

		for (int i = 0; i < 4; i++) {
			if ((bvhs[dataind].childFlag & (1 << i)) == (1 << i)) // is null
			{
				continue;
			}

			if (((bvhs[dataind].childFlag >> 4) & (1 << i)) == (1 << i)) // if leaf
			{
				int triIdx = bvhs[dataind].childs[i].triIdx;
				bool isHit = intersect(&rays[idx], vertices[trilist[triIdx].x], vertices[trilist[triIdx].y], vertices[trilist[triIdx].z]);
				if (isHit) {
					rays[idx].isHit = true;
				}
			}
			else {
				bool isHit = intersect(&rays[idx], bvhs[dataind], i);
				if (isHit) {
					nodeind++;
					stack[nodeind] = bvhs[dataind].childs[i].boxIdx;
				}
			}
		}
	}
}

std::vector<AISTER_GRAPHICS_ENGINE::RayHit> RayTraverse(glm::vec3 origin, std::vector<glm::vec3> dirlist, std::vector<AISTER_GRAPHICS_ENGINE::QBVH4Node> bvh, std::vector<glm::vec3> vertices, std::vector<AISTER_GRAPHICS_ENGINE::__Tri_t> trilist, int vsize, int tsize, int width, int height) {
	int arraysize = width * height;

	GPURay* rayGPU;
	cudaMalloc(&rayGPU, sizeof(GPURay) * arraysize);

	GPURay* cpuRay = new GPURay[arraysize];

	GPUQBVH4* bvhGPU;
	cudaMalloc(&bvhGPU, sizeof(GPUQBVH4) * bvh.size());
	cudaMemcpy(bvhGPU, &bvh[0], sizeof(GPUQBVH4) * bvh.size(), cudaMemcpyHostToDevice);

	float3* vert; int3* tri;
	cudaMalloc(&vert, sizeof(float3) * vertices.size());
	cudaMalloc(&tri, sizeof(int3) * trilist.size());
	cudaMemcpy(vert, &vertices[0], sizeof(float3) * vertices.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(tri, &trilist[0], sizeof(int3) * trilist.size(), cudaMemcpyHostToDevice);

	for (int i = 0; i < arraysize; i++) {
		cpuRay[i].origin = make_float3(origin.x, origin.y, origin.z);
		cpuRay[i].dir = make_float3(dirlist[i].x, dirlist[i].y, dirlist[i].z);
		cpuRay[i].tmin = 1e34f;
		cpuRay[i].tmax = 1e34f;
		cpuRay[i].isHit = false;
	}

	dim3 block_(16, 16);
	dim3 grid_(ceil(width / 16.f), ceil(height / 16.f));

	cudaMemcpy(rayGPU, cpuRay, sizeof(GPURay) * arraysize, cudaMemcpyHostToDevice);
	//
	traverse << <grid_, block_ >> > (rayGPU, bvhGPU, vert, tri, vertices.size(), trilist.size(), width, height);
	cudaDeviceSynchronize();
	//
	cudaMemcpy(cpuRay, rayGPU, sizeof(GPURay) * arraysize, cudaMemcpyDeviceToHost);

	cudaFree(bvhGPU);
	cudaFree(rayGPU);
	cudaFree(vert);
	cudaFree(tri);
	
	std::vector<AISTER_GRAPHICS_ENGINE::RayHit> res;
	for (int i = 0; i < arraysize; i++) {
		if (cpuRay[i].isHit) {
			float3 resultPos = (cpuRay[i].origin + cpuRay[i].dir * cpuRay[i].tmin);
			res.push_back(AISTER_GRAPHICS_ENGINE::RayHit(glm::vec3(resultPos.x, resultPos.y, resultPos.z), cpuRay[i].tmin));
		}
	}

	delete[] cpuRay;

	return res;
}