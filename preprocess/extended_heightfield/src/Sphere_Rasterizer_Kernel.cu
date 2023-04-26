#pragma once

#include "Sphere_Rasterizer_Kernel.h"

#include <iostream>
#include <stdexcept>

/* The rasterizer performs z-buffer rasterization by brute force looping all primitives. 
*/

#define empty 65535.0f
#define empty_interval make_float2( empty, empty )

void throw_on_cuda_error()
{
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		throw std::runtime_error(cudaGetErrorString(error));
	}
};

__global__ void rasterize_sphere_kernel(Sphere* spheres,
								        int n_spheres,
								        float2* extended_heightfield,
	                                    int2 output_resolution,
										int n_hf_entries,
										float image_plane_z )
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx >= output_resolution.x)
		return;
	if (idx >= output_resolution.y)
		return;

	int pixel_index = idx * output_resolution.y + idy;

	// initialize extended hf
	for (int i = 0; i < n_hf_entries; i++)
		extended_heightfield[pixel_index * n_hf_entries + i] = empty_interval;

	const float pixel_x = (float) idx;
	const float pixel_y = (float) idy;
	// loop over all spheres
	for (int sphere_id = 0; sphere_id < n_spheres; sphere_id++)
	{
		const Sphere& sphere = spheres[sphere_id];
		const float dz = fabsf( sphere.z - image_plane_z);
		if ( dz >= sphere.r )
			continue;

		// early termination if the pixel coordinates are outside the sphere
		const float r = sqrtf( sphere.r * sphere.r - dz * dz );
		if ( pixel_x < sphere.x - r || pixel_x > sphere.x + r )
			continue;
		if ( pixel_y < sphere.x - r || pixel_x > sphere.x + r )
			continue;

		// calculate entry and exit point by computing both solutions to r^2 = (x-x0)^2 + (y-y0)^2 + (z-z0)^2
		const float dx = pixel_x - sphere.x;
		const float dy = pixel_y - sphere.y;

		const float square_term = sqrtf( sphere.r * sphere.r - dx * dx - dy * dy );
		float entry = sphere.z - square_term;
		float exit  = sphere.z + square_term;

		// handle the case that the sphere is cut by the image place 
		if (entry < image_plane_z)
			entry = image_plane_z;

		unsigned int i = 0;
		while (extended_heightfield[i].x < entry)
			i++;
		float2 replaced_number = extended_heightfield[i];
		extended_heightfield[i] = make_float2( entry, exit );
		i++;
		do
		{
			float2 tmp = extended_heightfield[i];
			extended_heightfield[i] = replaced_number;
			replaced_number = tmp;
		} while (replaced_number.x < empty);

	}
}

/*Sphere_Rasterizer_Kernel::Sphere_Rasterizer_Kernel(std::vector<Sphere>& spheres, int2 output_resolution, int n_hf_entries)
	: spheres_cpu(spheres)
	, output_resolution(output_resolution)
	, n_hf_entries(n_hf_entries)
{
	spheres_gpu = allocate_spheres_on_gpu(spheres_cpu);
	extended_heightfield_gpu = allocate_extended_heightfield_on_gpu();
} */

Sphere_Rasterizer_Kernel::Sphere_Rasterizer_Kernel(py::array& spheres, std::pair<int, int> output_resolution, int n_hf_entries) // , std::pair<int, int> output_resolution, 
	// : output_resolution( make_int2( std::get<0>(output_resolution), std::get<1>(output_resolution) ) )
	// , n_hf_entries(n_hf_entries)
{
	std::cout << "creating extended heightfield of resolution " << std::get<0>(output_resolution) << "/" << std::get<1>(output_resolution) << std::endl;
	allocate_spheres_cpu(spheres);
	spheres_gpu = allocate_spheres_on_gpu(spheres_cpu);
	extended_heightfield_gpu = allocate_extended_heightfield_on_gpu();
}

Sphere_Rasterizer_Kernel::~Sphere_Rasterizer_Kernel()
{
}

std::vector<std::tuple<float, float>> Sphere_Rasterizer_Kernel::rasterize_spheres( float image_plane )
{
	return extended_heightfield_cpu;
}

void Sphere_Rasterizer_Kernel::allocate_spheres_cpu(py::array& spheres)
{
	py::buffer_info info = spheres.request();
	if (info.ndim != 2)
		throw std::invalid_argument("spheres array is expected to be of dimensions nx4");
	if (info.shape[1] != 4)
		throw std::invalid_argument("spheres array is expected to be of dimensions nx4");
	spheres_cpu.resize(info.shape[0]);
	double* ptr = (double*) info.ptr;
	for (size_t i = 0; i < info.shape[0]; i++)
	{
		spheres_cpu[i].x = *(ptr++);
		spheres_cpu[i].y = *(ptr++);
		spheres_cpu[i].z = *(ptr++);
		spheres_cpu[i].r = *(ptr++);
	}
	for (auto it : spheres_cpu)
		std::cout << "Sphere: " << it.x << " " << it.y << " " << it.z << " / " << it.r << std::endl;
}

void Sphere_Rasterizer_Kernel::call_kernel()
{
	int2 grid_size = output_resolution;
	dim3 block_size(32, 32);
	dim3 num_blocks((grid_size.x + block_size.x - 1) / block_size.x, (grid_size.y + block_size.y - 1) / block_size.y );
	rasterize_sphere_kernel << <num_blocks, block_size >> > (spheres_gpu, spheres_cpu.size(), extended_heightfield_gpu, output_resolution, n_hf_entries, image_plane);
	throw_on_cuda_error();
}

Sphere* Sphere_Rasterizer_Kernel::allocate_spheres_on_gpu( const std::vector<Sphere>& spheres_cpu )
{
	int n_spheres = (int) spheres_cpu.size();
	Sphere* ptr_gpu;
	cudaMalloc((void**)&ptr_gpu, sizeof(Sphere) * n_spheres);
	cudaMemcpy(ptr_gpu, &spheres_cpu[0], sizeof(Sphere) * n_spheres, cudaMemcpyHostToDevice);
	return ptr_gpu;
}

float2* Sphere_Rasterizer_Kernel::allocate_extended_heightfield_on_gpu()
{
	float2* ptr_gpu;
	cudaMalloc((void**)&ptr_gpu, sizeof(float2) * output_resolution.x * output_resolution.y * n_hf_entries);
	return ptr_gpu;
}