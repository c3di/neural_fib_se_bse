#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "python_utils.h"

#include <vector>
#include <tuple>

template<class Primitive>
class Abstract_Rasterizer
{
public:
	Abstract_Rasterizer(py::array& primitives, std::pair<int, int> output_resolution, int n_hf_entries, int max_buffer_length = 64);
	Abstract_Rasterizer(float2* extended_heightfield_gpu, py::array& primitives, std::pair<int, int> output_resolution, int n_hf_entries, int max_buffer_length = 64);
	Abstract_Rasterizer(float2* extended_heightfield_gpu, std::vector<Primitive>& primitives, std::pair<int, int> output_resolution, int n_hf_entries, int max_buffer_length = 64);
	virtual ~Abstract_Rasterizer();

	virtual std::pair< py::array_t<float>, py::array_t<float> > rasterize_py( float image_plane );
	virtual void rasterize( float image_plane ) = 0;

	virtual py::array_t<float> get_normal_map_py();
	virtual std::vector<float> get_normal_map();
	virtual py::array_t<float> get_extended_height_field_py();

protected:
	virtual void allocate_primitives_cpu(py::array& spheres);

	virtual Primitive* allocate_primitives_on_gpu(const std::vector<Primitive>& primitives_cpu);

	virtual void presort_primitives();

protected:
	std::vector<Primitive> primitives_cpu;
	Primitive* primitives_gpu;
	int n_primitives;

	float2* extended_heightfield_gpu;
	float3* normal_map_gpu;
	float*  z_buffer_gpu;

	int2 output_resolution;
	int n_hf_entries;
	int buffer_length;
	float image_plane;
};
