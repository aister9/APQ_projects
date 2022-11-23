#pragma once

#include "optix7.h"
#include "CUDABuffer.h"

struct custom_cuda_groups {
	osc::CUDABuffer __a;
	osc::CUDABuffer __b;
	osc::CUDABuffer __c;
};

void addCall(custom_cuda_groups& tt);
void generate(custom_cuda_groups& tt);

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