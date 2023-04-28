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
	// Sphere_Rasterizer_Kernel(std::vector<Sphere>& spheres, int2 output_resolution, int n_hf_entries);
	Sphere_Rasterizer(py::array& spheres, std::pair<int, int> output_resolution, int n_hf_entries, int max_buffer_length = 64);
	Sphere_Rasterizer(float2* extended_heightfield_gpu, py::array& spheres, std::pair<int, int> output_resolution, int n_hf_entries, int max_buffer_length = 64);
	~Sphere_Rasterizer();

	py::array_t<float> rasterize_spheres_py( float image_plane );
	void rasterize_spheres( float image_plane );

protected:
	void allocate_spheres_cpu(py::array& spheres);

	Sphere* allocate_spheres_on_gpu(const std::vector<Sphere>& spheres_cpu);
	float2* allocate_extended_heightfield_on_gpu();

	void presort_spheres();

protected:
	std::vector<Sphere> spheres_cpu;
	Sphere* spheres_gpu;
	int n_spheres;

	py::array_t<float> extended_heightfield_cpu;
	float2* extended_heightfield_gpu;

	int2 output_resolution;
	int n_hf_entries;
	int buffer_length;
	float image_plane;
};
