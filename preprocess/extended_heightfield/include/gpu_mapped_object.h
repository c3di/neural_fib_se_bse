#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;

template<typename DTYPE>
class GPUMappedObject
{
public:
	GPUMappedObject<DTYPE>(int3 dimensions);
	GPUMappedObject<DTYPE>(int3 dimensions, DTYPE init_value);
	GPUMappedObject<DTYPE>(int3 dimensions, DTYPE* gpu_ptr);
	GPUMappedObject<DTYPE>(int3 dimensions, DTYPE* cpu_ptr, DTYPE* gpu_ptr);

	virtual ~GPUMappedObject<DTYPE>();

	void push_on_gpu();
	void pull_from_gpu();

	DTYPE* cpu_ptr();
	DTYPE* gpu_ptr();
	// std::vector<DTYPE>& as_cpp();
	py::array_t<DTYPE> as_py();

	// void resize(int3 newDimensions);
	void set_cpu(DTYPE* cpu_ptr);

protected:
	void call_mem_set_kernel(DTYPE init_value);

public:
	DTYPE* _cpu_ptr;
	DTYPE* _gpu_ptr;

	bool ownsCPUBuffer;
	bool ownsGPUBuffer;
	int3 dimensions;
};

typedef GPUMappedObject<float>  GPUMappedFloatBuffer;
typedef GPUMappedObject<float2> GPUMappedFloat2Buffer;
typedef GPUMappedObject<float3> GPUMappedFloat3Buffer;