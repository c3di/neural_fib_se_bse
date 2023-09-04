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

template<typename T>
T* allocate_buffer_on_gpu(int3 buffer_size)
{
	T* ptr_gpu;
	cudaMalloc((void**)&ptr_gpu, sizeof(T) * buffer_size.x * buffer_size.y * buffer_size.z);
	return ptr_gpu;
}

template<typename T>
__global__ void mem_set_kernel(T* buffer, int3 buffer_size, T init_value)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int idz = blockIdx.z * blockDim.z + threadIdx.z;

	if (idx >= buffer_size.x)
		return;
	if (idy >= buffer_size.y)
		return;
	if (idz >= buffer_size.z)
		return;

	int pixel_index = idz * buffer_size.x * buffer_size.y + idy * buffer_size.x + idx;

	buffer[pixel_index] = init_value;
}

template<typename T>
__host__ void call_mem_set_kernel(T* buffer, int3 buffer_size, T init_value)
{
	dim3 block_size(32, 32, 1);
	dim3 num_blocks((buffer_size.x + block_size.x - 1) / block_size.x, (buffer_size.y + block_size.y - 1) / block_size.y, (buffer_size.z + block_size.z - 1) / block_size.z);
	mem_set_kernel<T> << <num_blocks, block_size >> > (buffer, buffer_size, init_value );
	throw_on_cuda_error();
}

template<typename T>
__host__ T* allocate_buffer_on_gpu(int3 buffer_size, T init_value)
{
	T* buffer = allocate_buffer_on_gpu<T>(buffer_size);
	call_mem_set_kernel( buffer, buffer_size, init_value );
	return buffer;
}

template float*  allocate_buffer_on_gpu<float >(int3 buffer_size);
template float* allocate_buffer_on_gpu<float >(int3 buffer_size, float);

template float2* allocate_buffer_on_gpu<float2>(int3 buffer_size);
template float2* allocate_buffer_on_gpu<float2>(int3 buffer_size, float2);

template float3* allocate_buffer_on_gpu<float3>(int3 buffer_size);
template float3* allocate_buffer_on_gpu<float3>(int3 buffer_size, float3);

template __global__ void mem_set_kernel(float* buffer, int3 buffer_size, float init_value);
template __global__ void mem_set_kernel(float2* buffer, int3 buffer_size, float2 init_value);
template __global__ void mem_set_kernel(float3* buffer, int3 buffer_size, float3 init_value);