
#include <cuda.h>

#include "MyCudaAdd.h"
#include <curand.h>

__global__ void addKernel(int* a, int* b, int* c, size_t size);
__global__ void setupKernel(int* a, int* b, int* c, size_t size);

void generate(custom_cuda_groups &tt) {
	tt.__a.alloc(1024 * 1024 * sizeof(int));
	tt.__b.alloc(1024 * 1024 * sizeof(int));
	tt.__c.alloc(1024 * 1024 * sizeof(int));

	setupKernel << <1024, 1024 >> > ((int*)tt.__a.d_pointer(), (int*)tt.__b.d_pointer(), (int*)tt.__c.d_pointer(), 1024*1024);
}

void addCall(custom_cuda_groups& tt) {
	addKernel << <1024, 1024 >> > ((int*)tt.__a.d_pointer(), (int*)tt.__b.d_pointer(), (int*)tt.__c.d_pointer(), 1024 * 1024);
	CUDA_SYNC_CHECK();
}


__global__ void setupKernel(int* a, int* b, int* c, size_t size) {

	const int tidx = blockIdx.x * blockDim.x + threadIdx.x;

	if (tidx >= size) return;

	a[tidx] = tidx;
	b[tidx] = tidx * 2;
	c[tidx] = 0;
}

__global__ void addKernel(int* a, int* b, int* c, size_t size) {

	const int tidx = blockIdx.x * blockDim.x + threadIdx.x;

	if (tidx >= size) return;

	c[tidx] = a[tidx] + b[tidx];
}