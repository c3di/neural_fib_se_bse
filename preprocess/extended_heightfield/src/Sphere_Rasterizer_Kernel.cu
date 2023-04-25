#pragma once

#include "sphere.h"
#include "Sphere_Rasterizer_Kernel.h"

/* The rasterizer performs z-buffer rasterization by brute force looping all primitives. 
*/

__global__ void rasterize_sphere(Sphere* spheres,
								 int n_spheres,
								 unsigned short* extended_heightfield,
								 unsigned char* normal_map,
								 int n_heightfield_channels,
	                             int2 output_resolution)
{
}

Sphere_Rasterizer_Kernel::Sphere_Rasterizer_Kernel( std::vector<Sphere> spheres, int2 output_resolution, int n_hf_entries)
: spheres_cpu(spheres)
, output_resolution(output_resolution)
, n_hf_entries(n_hf_entries)
{
	spheres_gpu = allocate_spheres_on_gpu(spheres_cpu);
	extended_heightfield_gpu = allocate_extended_heightfield_on_gpu();
}

Sphere_Rasterizer_Kernel::~Sphere_Rasterizer_Kernel()
{
}

Sphere* Sphere_Rasterizer_Kernel::allocate_spheres_on_gpu( const std::vector<Sphere>& spheres_cpu )
{
	int n_spheres = (int) spheres_cpu.size();
	Sphere* ptr_gpu;
	cudaMalloc((void**)&ptr_gpu, sizeof(Sphere) * n_spheres);
	cudaMemcpy(ptr_gpu, &spheres_cpu[0], sizeof(Sphere) * n_spheres, cudaMemcpyHostToDevice);
	return ptr_gpu;
}

unsigned short* Sphere_Rasterizer_Kernel::allocate_extended_heightfield_on_gpu()
{
	unsigned short* ptr_gpu;
	cudaMalloc((void**)&ptr_gpu, sizeof(unsigned short) * output_resolution.x * output_resolution.y * n_hf_entries);
	return ptr_gpu;
}