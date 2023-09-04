#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <stdexcept>

#include "gpu_mapped_object.h"
#include "cuda_utils.h"

template<typename DTYPE>
__global__ void mem_set_kernel(DTYPE* buffer, int3 buffer_size, DTYPE init_value)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int idz = blockIdx.z * blockDim.z + threadIdx.z;

	if (idx >= buffer_size.x)
		return;
	if (idy >= buffer_size.y)
		return;

	int pixel_index = idz * buffer_size.x * buffer_size.y + idy * buffer_size.x + idx;

	buffer[pixel_index] = init_value;
}

template<typename DTYPE>
GPUMappedObject<DTYPE>::GPUMappedObject<DTYPE>(int3 dimensions)
	: ownsCPUBuffer(true)
	, ownsGPUBuffer(true)
	, dimensions(dimensions)
{
	_gpu_ptr = allocate_buffer_on_gpu<DTYPE>(dimensions);
	cudaMallocHost( &_cpu_ptr, sizeof(DTYPE) * dimensions.x * dimensions.y * dimensions.z );
}

template<typename DTYPE>
GPUMappedObject<DTYPE>::GPUMappedObject<DTYPE>(int3 dimensions, DTYPE init_value)
	: ownsCPUBuffer(true)
	, ownsGPUBuffer(true)
	, dimensions(dimensions)
{
	_gpu_ptr = allocate_buffer_on_gpu<DTYPE>(dimensions);
	call_mem_set_kernel(init_value);
	cudaMallocHost(&_cpu_ptr, sizeof(DTYPE) * dimensions.x * dimensions.y * dimensions.z);
	for ( size_t i = 0; i < dimensions.x * dimensions.y * dimensions.z; i++ )
		*(_cpu_ptr + i) = init_value;
}

template<typename DTYPE>
GPUMappedObject<DTYPE>::GPUMappedObject<DTYPE>(int3 dimensions, DTYPE* gpu_ptr)
	: ownsCPUBuffer(true)
	, ownsGPUBuffer(false)
	, dimensions(dimensions)
	, _gpu_ptr(gpu_ptr)
{
	cudaMallocHost(&_cpu_ptr, sizeof(DTYPE) * dimensions.x * dimensions.y * dimensions.z);
}

template<typename DTYPE>
GPUMappedObject<DTYPE>::GPUMappedObject<DTYPE>(int3 dimensions, DTYPE* cpu_ptr, DTYPE* gpu_ptr )
	: _cpu_ptr(cpu_ptr)
	, _gpu_ptr(gpu_ptr)
	, dimensions(dimensions)
	, ownsCPUBuffer(false)
	, ownsGPUBuffer(false)
{
}

template<typename DTYPE>
GPUMappedObject<DTYPE>::~GPUMappedObject<DTYPE>()
{
	if (ownsCPUBuffer)
		cudaFreeHost(_cpu_ptr);
	if (ownsGPUBuffer)
		cudaFree(_gpu_ptr);
}

template<typename DTYPE>
void GPUMappedObject<DTYPE>::push_on_gpu()
{
	cudaMemcpy((void*) _gpu_ptr, (void*) _cpu_ptr, sizeof(DTYPE) * dimensions.x * dimensions.y * dimensions.z, cudaMemcpyHostToDevice);
}

template<typename DTYPE>
void GPUMappedObject<DTYPE>::pull_from_gpu()
{
	size_t size_in_bytes = sizeof(DTYPE) * dimensions.x * dimensions.y * dimensions.z;
	cudaMemcpy((void*)_cpu_ptr, (void*) _gpu_ptr, size_in_bytes, cudaMemcpyDeviceToHost);
}

template<typename DTYPE>
DTYPE* GPUMappedObject<DTYPE>::cpu_ptr()
{
	return _cpu_ptr;
}

template<typename DTYPE>
DTYPE* GPUMappedObject<DTYPE>::gpu_ptr()
{
	return _gpu_ptr;
}

/* template<typename DTYPE>
std::vector<DTYPE>& GPUMappedObject<DTYPE>::as_cpp()
{
	if (!ownsCPUBuffer)
		throw std::runtime_error("cannot access std::vector<DTYPE> representation of buffer that is owned by other entity");
	return buffer;
} */

template<typename DTYPE>
py::array_t<DTYPE> GPUMappedObject<DTYPE>::as_py()
{
	return py::array(
		py::buffer_info(
			_cpu_ptr,																					  /* Pointer to data */
			sizeof(DTYPE),																				  /* Size of one item */
			py::format_descriptor<DTYPE>::format(),   				                                      /* Buffer format */
			3,																							  /* How many dimensions? */
			{ dimensions.x, dimensions.y, dimensions.z },												  /* Number of elements for each dimension */
			{ dimensions.y * dimensions.z * sizeof(DTYPE), dimensions.z * sizeof(DTYPE), sizeof(DTYPE) }  /* Strides for each dimension */
		)
	);
}

template<typename DTYPE>
void GPUMappedObject<DTYPE>::set_cpu(DTYPE* cpu_ptr)
{
	if ( ownsCPUBuffer )
		cudaFreeHost(_cpu_ptr);
	_cpu_ptr = cpu_ptr;
	ownsCPUBuffer = false;
}

template<typename DTYPE>
__host__ void GPUMappedObject<DTYPE>::call_mem_set_kernel(DTYPE init_value)
{
	dim3 block_size(32, 32, 1);
	dim3 num_blocks((dimensions.x + block_size.x - 1) / block_size.x, (dimensions.y + block_size.y - 1) / block_size.y, (dimensions.z + block_size.z - 1) / block_size.z);
	mem_set_kernel<DTYPE> << <num_blocks, block_size >> > (_gpu_ptr, dimensions, init_value);
	throw_on_cuda_error();
}


template class GPUMappedObject<float>;
template class GPUMappedObject<float2>;
template class GPUMappedObject<float3>;
