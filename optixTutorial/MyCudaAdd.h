#pragma once

#include "optix7.h"
#include "CUDABuffer.h"

#include <iostream>

struct custom_cuda_groups {
	osc::CUDABuffer __a;
	osc::CUDABuffer __b;
	osc::CUDABuffer __c;
};

void addCall(custom_cuda_groups& tt);
void generate(custom_cuda_groups& tt);

void customCudaCheck();