
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


void customCudaCheck() {
    //custom cuda check
    int* cu_a, * cu_b, * cu_c;
    cu_a = new int[1024 * 1024];
    cu_b = new int[1024 * 1024];
    cu_c = new int[1024 * 1024];

    custom_cuda_groups tt;
    std::cout << "array generate" << std::endl;
    generate(tt); // generate
    std::cout << "array addition" << std::endl;
    addCall(tt); // addcall

    std::cout << "array download" << std::endl;
    tt.__a.download(cu_a, 1024 * 1024);
    tt.__b.download(cu_b, 1024 * 1024);
    tt.__c.download(cu_c, 1024 * 1024);

    std::cout << "check result" << std::endl;
    for (int i = 0; i < 1024 * 1024; i++) {
        if (cu_c[i] != cu_a[i] + cu_b[i]) {
            std::cout << cu_c[i] << "!=" << cu_a[i] << "+" << cu_b[i] << std::endl;
            break;
        }
    }
    std::cout << "complete" << std::endl;
}