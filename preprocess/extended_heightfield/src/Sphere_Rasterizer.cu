#pragma once

#include "Sphere_Rasterizer.h"

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

__global__ void resolve_csg_kernel(
	float2* extended_heightfield,
	int3 output_resolution,
	int n_hf_entries)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx >= output_resolution.x)
		return;
	if (idx >= output_resolution.y)
		return;

	int pixel_index = idx * output_resolution.y + idy;

	int i = output_resolution.z;
	// while (extended_heightfield[pixel_index * output_resolution.z + i].x != empty) 
	while ( i > 0 )
	{
		for (int j = 0; j < i-1; j++)
		{
			if (idx == 8 && idy == 4)
			{
				for (int ii = 0; ii < output_resolution.z; ii++)
					printf("    %i : %.2f %.2f\n", ii, extended_heightfield[pixel_index * output_resolution.z + ii].x, extended_heightfield[pixel_index * output_resolution.z + ii].y);
			}

			float2& a = extended_heightfield[pixel_index * output_resolution.z + j];
			float2& b = extended_heightfield[pixel_index * output_resolution.z + j + 1];

			// case A: order ok
			if (a.y < b.x)
			{
				if (idx == 8 && idy == 4)
					printf("  %i %i noop\n", i, j );
			}

			// case B: swap
			else if (a.x > b.y)
			{
				float2 tmp = b;
				b = a;
				a = tmp;
				if (idx == 8 && idy == 4)
					printf("  %i %i swap %.2f %.2f <-> %.2f %.2f \n", i, j, a.x, a.y, b.x, b.y );
			}

			// case C: merge
			else {
				float new_x = fminf(a.x, b.x);
				float new_y = fmaxf(a.y, b.y);
				if (idx == 8 && idy == 4) 
				{
					printf("  %i %i merge %.2f %.2f <-> %.2f %.2f \n", i, j, a.x, a.y, b.x, b.y);
				}
				a = make_float2(new_x, new_y);
				b = empty_interval;
				continue;
			} // merge

		}
		i--;
	}
}

__global__ void rasterize_sphere_kernel(Sphere* spheres,
								        int n_spheres,
								        float2* extended_heightfield, // contains entry/exit information as float2 per pixel
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
	for (int i = 0; i < n_spheres; i++)
		extended_heightfield[pixel_index * n_spheres + i] = empty_interval;

	const float pixel_x = (float) idx;
	const float pixel_y = (float) idy;

	int hit_index = 0;

	// loop over all spheres
	for (int sphere_id = 0; sphere_id < n_spheres; sphere_id++)
	{
		const Sphere& sphere = spheres[sphere_id];

		const float dz = fabsf( sphere.z - image_plane_z);

		// early termination if sphere behind image plane
		if ( dz <= -sphere.r )
			continue;

		// calculate entry and exit point by computing both solutions to r^2 = (x-x0)^2 + (y-y0)^2 + (z-z0)^2
		const float dx = pixel_x - sphere.x;
		const float dy = pixel_y - sphere.y;

		// check if intersection point exists
		if (dx * dx + dy * dy > sphere.r * sphere.r)
			continue;

		const float square_term = sqrtf( sphere.r * sphere.r - dx * dx - dy * dy );
		float entry = sphere.z - square_term;
		float exit  = sphere.z + square_term;

		// handle the case that the sphere is cut by the image place 
		if (entry < image_plane_z)
			entry = image_plane_z;

		if (idx == 8 && idy == 4)
			printf("  creating %.2f %.2f\n", entry, exit );

		extended_heightfield[pixel_index * n_spheres + hit_index] = make_float2( entry, exit );
		hit_index++;
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

Sphere_Rasterizer::Sphere_Rasterizer(py::array& spheres, std::pair<int, int> output_resolution, int n_hf_entries) // , std::pair<int, int> output_resolution, 
	: output_resolution( make_int2( std::get<0>(output_resolution), std::get<1>(output_resolution) ) )
	, n_hf_entries(n_hf_entries)
{
	std::cout << "creating extended heightfield of resolution " << std::get<0>(output_resolution) << "/" << std::get<1>(output_resolution) << std::endl;
	allocate_spheres_cpu(spheres);
	spheres_gpu = allocate_spheres_on_gpu(spheres_cpu);
	extended_heightfield_gpu = allocate_extended_heightfield_on_gpu();
}

Sphere_Rasterizer::~Sphere_Rasterizer()
{
}

py::array_t<float> Sphere_Rasterizer::rasterize_spheres_py( float image_plane )
{
	rasterize_spheres( image_plane );
	auto pyarray = create_py_array(output_resolution.x, output_resolution.y, n_spheres * 2);
	cudaMemcpy(pyarray.request().ptr, extended_heightfield_gpu, sizeof(float2) * output_resolution.x * output_resolution.y * n_spheres, cudaMemcpyDeviceToHost );
	return pyarray;
}

void Sphere_Rasterizer::rasterize_spheres( float image_plane )
{
	int2 grid_size = output_resolution;
	dim3 block_size(32, 32);
	dim3 num_blocks((grid_size.x + block_size.x - 1) / block_size.x, (grid_size.y + block_size.y - 1) / block_size.y);
	rasterize_sphere_kernel << <num_blocks, block_size >> > (spheres_gpu, spheres_cpu.size(), extended_heightfield_gpu, output_resolution, n_hf_entries, image_plane);

	int3 buffer_size = make_int3(output_resolution.x, output_resolution.y, n_spheres);
	resolve_csg_kernel << <num_blocks, block_size >> > (extended_heightfield_gpu, buffer_size, n_hf_entries);
	throw_on_cuda_error();
}

py::array_t<float> Sphere_Rasterizer::create_py_array( int shape0, int shape1, int shape2 )
{
	return py::array(py::buffer_info(
			         nullptr,                                                                   /* Pointer to data (nullptr -> ask NumPy to allocate!) */
                     sizeof(float),                                                              /* Size of one item */
                     py::format_descriptor<float>::value,                                        /* Buffer format */
                     3,																		     /* How many dimensions? */
                     { shape0, shape1, shape2 },                                                 /* Number of elements for each dimension */
                     { shape1 * shape2 * sizeof(float), shape2 * sizeof(float), sizeof(float) }  /* Strides for each dimension */
	));
}

void Sphere_Rasterizer::allocate_spheres_cpu(py::array& spheres)
{
py::buffer_info info = spheres.request();
	if (info.ndim != 2)
		throw std::invalid_argument("spheres array is expected to be of dimensions nx4");
	if (info.shape[1] != 4)
		throw std::invalid_argument("spheres array is expected to be of dimensions nx4");
	n_spheres = info.shape[0];
	spheres_cpu.resize(n_spheres);
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

Sphere* Sphere_Rasterizer::allocate_spheres_on_gpu( const std::vector<Sphere>& spheres_cpu )
{
	int n_spheres = (int) spheres_cpu.size();
	Sphere* ptr_gpu;
	cudaMalloc((void**)&ptr_gpu, sizeof(Sphere) * n_spheres);
	cudaMemcpy(ptr_gpu, &spheres_cpu[0], sizeof(Sphere) * n_spheres, cudaMemcpyHostToDevice);
	return ptr_gpu;
}

float2* Sphere_Rasterizer::allocate_extended_heightfield_on_gpu()
{
	float2* ptr_gpu;
	cudaMalloc((void**)&ptr_gpu, sizeof(float2) * output_resolution.x * output_resolution.y * n_spheres);
	return ptr_gpu;
}
