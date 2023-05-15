#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "python_utils.h"
#include "sphere.h"
#include "cylinder.h"

#include <vector>
#include <tuple>

class Intersector;
class Sphere_Intersector;
class Cylinder_Intersector;
class CSG_Resolver;

#ifdef extended_heightfield_EXPORTS
#define LIBRARY_API __declspec(dllexport)
#else
#define LIBRARY_API __declspec(dllimport)
#endif

class HeightFieldExtractor
{
public:
	HeightFieldExtractor(std::tuple<int, int> output_resolution, int n_hf_entries, int max_buffer_length = 64);
	~HeightFieldExtractor();

	void add_spheres_py(py::array& spheres);
	void add_spheres(std::vector<Sphere>& spheres);

	void add_cylinders_py(py::array& cylinders);
	void add_cylinders(std::vector<Cylinder>& cylinders);

	std::tuple< std::vector<float2>, std::vector<float3>> extract_data_representation(float image_plane);
	std::tuple< py::array_t<float2>, py::array_t<float3>> extract_data_representation_py(float image_plane);
	void intersect(float image_plane );

protected:
	py::array_t<float2> collect_extended_heightfield_py();
	std::vector<float2> collect_extended_heightfield();
	void call_result_collection_kernel();

protected:
	std::vector<Intersector*> intersectors;
	CSG_Resolver* csg_resolver;

	float2* extended_heightfield_gpu;
	float* z_buffer_gpu;
	float3* normal_map_gpu;

	float2* result_gpu;

	int2 output_resolution;
	int n_hf_entries;
	int max_buffer_length;
	float image_plane;
};
