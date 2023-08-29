#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <cuda.h>
#include <cuda_runtime.h>

#include "python_utils.h"
#include "sphere.h"
#include "cylinder.h"
#include "cuboid.h"

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

/* Die HeightFieldExtractor is the top-level interface for the functionality of the module. The usage is to
   create the HeightFieldExtractor, then add primitives, then call extract_data_representation. 
   The data representation is as pair of extended height-field and normalmap.
       1. The extended heightfield has dimensions res_x * res_y * n_hf_entries, where each hf_entry consists of a float2 of entry, and exit-point
       2. The normal-map has dimensions res_x * res_y, where each entry is a float3 of a normalized world-space vector (x/y/z)
 */
class HeightFieldExtractor
{
public:
	HeightFieldExtractor(std::tuple<int, int> output_resolution, int n_hf_entries, int max_buffer_length = 64);
	~HeightFieldExtractor();

	void add_spheres_py(py::array& spheres);
	void add_spheres(std::vector<Sphere>& spheres);

	void add_cylinders_py(py::array& cylinders);
	void add_cylinders(std::vector<Cylinder>& cylinders);

	void add_cuboids_py(py::array& cuboids);
	void add_cuboids(std::vector<Cuboid>& cuboids);

	std::tuple< float2*, float3* > extract_data_representation(float image_plane);
	std::tuple< py::array_t<float2>, py::array_t<float3>> extract_data_representation_py(float image_plane);
	void intersect(float image_plane );

protected:
	py::array_t<float2> collect_extended_heightfield_py();
	float2* collect_extended_heightfield();
	void call_result_collection_kernel();

protected:
	std::vector<Intersector*> intersectors;
	CSG_Resolver* csg_resolver;

	float2* extended_heightfield_gpu;
	float* z_buffer_gpu;
	float3* normal_map_gpu;

	float2* result_gpu;
	float2* result_cpu;

	int2 output_resolution;
	int n_hf_entries;
	int max_buffer_length;
	float image_plane;
};
