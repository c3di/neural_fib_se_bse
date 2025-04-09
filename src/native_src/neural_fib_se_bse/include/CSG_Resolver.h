#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <cuda.h>
#include <cuda_runtime.h>

#include <tuple>

class CSG_Resolver
{
public:
	CSG_Resolver(float2* extended_heightfield_gpu, int3 buffer_size, int n_hf_entries);
	CSG_Resolver(py::array_t<float> extended_heightfield, int n_hf_entries);
	~CSG_Resolver();

	py::array_t<float> resolve_csg_py(float image_plane);
	void resolve_csg( float image_plane );

	float* get_extended_heightfield_cpu( py::array_t<float> extended_heightfield );

protected:
	py::array_t<float>* extended_heightfield_py = nullptr;
	float2* extended_heightfield_gpu;
	float* extended_heightfield_cpu;
	int3 buffer_size;
	int n_hf_entries;
};
