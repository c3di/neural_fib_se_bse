#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "python_utils.h"

#include <vector>
#include <tuple>

#include "sphere.h"

class Sphere_Rasterizer
{
public:
	Sphere_Rasterizer(py::array& spheres, std::pair<int, int> output_resolution, int n_hf_entries, int max_buffer_length = 64);
	Sphere_Rasterizer(float2* extended_heightfield_gpu, py::array& spheres, std::pair<int, int> output_resolution, int n_hf_entries, int max_buffer_length = 64);
	Sphere_Rasterizer(float2* extended_heightfield_gpu, std::vector<Sphere>& spheres, std::pair<int, int> output_resolution, int n_hf_entries, int max_buffer_length = 64);
	~Sphere_Rasterizer();

	std::pair< py::array_t<float>, py::array_t<float> > rasterize_spheres_py( float image_plane );
	void rasterize_spheres( float image_plane );

	py::array_t<float> get_normal_map_py();
	std::vector<float> get_normal_map();
	py::array_t<float> get_extended_height_field_py();

protected:
	void allocate_spheres_cpu(py::array& spheres);

	Sphere* allocate_spheres_on_gpu(const std::vector<Sphere>& spheres_cpu);

	void presort_spheres();

protected:
	std::vector<Sphere> spheres_cpu;
	Sphere* spheres_gpu;
	int n_spheres;

	float2* extended_heightfield_gpu;
	float3* normal_map_gpu;
	float*  z_buffer_gpu;

	int2 output_resolution;
	int n_hf_entries;
	int buffer_length;
	float image_plane;
};
