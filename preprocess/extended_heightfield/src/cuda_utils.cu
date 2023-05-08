#pragma once

#include <stdexcept>

#include "cuda_utils.h"

void throw_on_cuda_error()
{
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		throw std::runtime_error(cudaGetErrorString(error));
	}
};

float* allocate_float_buffer_on_gpu(int3 buffer_size)
{
	float* ptr_gpu;
	cudaMalloc((void**)&ptr_gpu, sizeof(float) * buffer_size.x * buffer_size.y * buffer_size.z);
	return ptr_gpu;
}

float2* allocate_float2_buffer_on_gpu( int3 buffer_size )
{
	float2* ptr_gpu;
	cudaMalloc((void**)&ptr_gpu, sizeof(float2) * buffer_size.x * buffer_size.y * buffer_size.z );
	return ptr_gpu;
}

float3* allocate_float3_buffer_on_gpu(int3 buffer_size)
{
	float3* ptr_gpu;
	cudaMalloc((void**)&ptr_gpu, sizeof(float3) * buffer_size.x * buffer_size.y * buffer_size.z);
	return ptr_gpu;
}