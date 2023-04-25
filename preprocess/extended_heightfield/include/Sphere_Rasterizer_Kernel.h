#pragma once

#include <vector>

#include "sphere.h"

class Sphere_Rasterizer_Kernel
{
public:
	Sphere_Rasterizer_Kernel(std::vector<Sphere> spheres, int2 output_resolution, int n_hf_entries);
	~Sphere_Rasterizer_Kernel();

protected:
	Sphere* allocate_spheres_on_gpu(const std::vector<Sphere>& spheres_cpu);
	unsigned short* allocate_extended_heightfield_on_gpu();

protected:
	std::vector<Sphere> spheres_cpu;
	Sphere* spheres_gpu;

	int2 output_resolution;
	int n_hf_entries;
	unsigned short* extended_heightfield_gpu;
};
