#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "python_utils.h"
#include "sphere.h"

#include <vector>
#include <tuple>

class Sphere_Rasterizer;
class CSG_Resolver;

#ifdef extended_heightfield_EXPORTS
#define LIBRARY_API __declspec(dllexport)
#else
#define LIBRARY_API __declspec(dllimport)
#endif

class HeightFieldExtractor
{
public:
	HeightFieldExtractor(py::array& spheres,           std::pair<int, int> output_resolution, int n_hf_entries, int max_buffer_length = 64);
	HeightFieldExtractor(std::vector<Sphere>& spheres, std::pair<int, int> output_resolution, int n_hf_entries, int max_buffer_length = 64);
	~HeightFieldExtractor();

	std::pair< std::vector<float>, std::vector<float>> extract_data_representation(float image_plane);
	std::pair< py::array_t<float>, py::array_t<float>> extract_data_representation_py(float image_plane);
	void rasterize(float image_plane );

protected:
	py::array_t<float> collect_extended_heightfield_py();
	std::vector<float> collect_extended_heightfield();
	void call_result_collection_kernel();

protected:
	Sphere_Rasterizer* sphere_rasterizer;
	CSG_Resolver* csg_resolver;

	float2* extended_heightfield_gpu;

	float2* result_gpu;

	int2 output_resolution;
	int n_hf_entries;
	int buffer_length;
	float image_plane;
};
