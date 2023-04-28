#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "python_utils.h"

#include <vector>
#include <tuple>

class Sphere_Rasterizer;
class CSG_Resolver;

class HeightFieldExtractor
{
public:
	HeightFieldExtractor(py::array& spheres, std::pair<int, int> output_resolution, int n_hf_entries, int max_buffer_length = 64 );
	~HeightFieldExtractor();

	py::array_t<float> rasterize_py( float image_plane );
	void rasterize(float image_plane );

protected:
	float2* allocate_extended_heightfield_on_gpu();

protected:
	Sphere_Rasterizer* sphere_rasterizer;
	CSG_Resolver* csg_resolver;

	py::array extended_heightfield_cpu;
	float2* extended_heightfield_gpu;

	int2 output_resolution;
	int n_hf_entries;
	int buffer_length;
	float image_plane;
};
