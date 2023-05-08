#pragma once

#include "Sphere_Rasterizer.h"

#include "cuda_utils.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>

/* The rasterizer performs z-buffer rasterization by brute force looping all primitives. 
*/
__global__ void rasterize_sphere_kernel(Sphere* spheres,
								        int n_spheres,
								        float2* extended_heightfield, // contains entry/exit information as float2 per pixel
									    float3* normal_map,
									    float* z_buffer,
	                                    int2 output_resolution,
										int buffer_length,
										int n_hf_entries,
										float image_plane_z,
										bool debug )
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx >= output_resolution.x)
		return;
	if (idy >= output_resolution.y)
		return;

	int pixel_index = idx * output_resolution.y + idy;

	// initialize extended hf
	for (int i = 0; i < buffer_length; i++)
		extended_heightfield[pixel_index * buffer_length + i] = empty_interval;

	// initialize z_buffer
	z_buffer[pixel_index] = empty;

	const float pixel_x = (float) idx;
	const float pixel_y = (float) idy;

	int hit_index = 0;

	// loop over all spheres
	for (int sphere_id = 0; sphere_id < n_spheres; sphere_id++)
	{
		const Sphere& sphere = spheres[sphere_id];

		if (debug && idx == 74 && idy == 45)
			printf("  %i : %.2f %.2f %.2f radius %.2f\n", sphere_id, sphere.x, sphere.y, sphere.z, sphere.r);

		const float dz = fabsf( sphere.z - image_plane_z);

		// early termination if sphere behind image plane
		if ( dz <= -sphere.r )
			continue;

		if (debug && idx == 74 && idy == 45)
			printf("    : front of image plance\n");

		// calculate entry and exit point by computing both solutions to r^2 = (x-x0)^2 + (y-y0)^2 + (z-z0)^2
		const float dx = pixel_x - sphere.x;
		const float dy = pixel_y - sphere.y;

		// check if intersection point exists
		if (dx * dx + dy * dy > sphere.r * sphere.r)
			continue;

		const float square_term = sqrtf( sphere.r * sphere.r - dx * dx - dy * dy );
		float entry = sphere.z - square_term;
		float exit  = sphere.z + square_term;

		bool cut_case = false;
		// handle the case that the sphere is cut by the image place 
		if (entry < image_plane_z)
		{
			entry = image_plane_z;
			cut_case = true;
		}

		if (debug && idx == 74 && idy == 45)
			printf("    : intersect %.2f - %.2f\n", entry, exit);

		extended_heightfield[pixel_index * buffer_length + hit_index] = make_float2( entry, exit );
		hit_index++;

		// write the normal map
		if (entry < z_buffer[pixel_index])
		{
			z_buffer[pixel_index] = entry;
			if (cut_case)
			{
				normal_map[pixel_index] = make_float3(0.0f, 0.0f, 1.0f);
			}
			else
			{
				const float xn = dx / sphere.r;
				const float yn = dy / sphere.r;
				const float zn = sqrtf(1.0f - xn * xn - yn * yn);
				if (debug && idx == 74 && idy == 45)
					printf("    : normal %.2f %.2f %.2f\n", xn, yn, zn);
				normal_map[pixel_index] = make_float3( xn,  yn, zn);
			}
		}

		if (hit_index > buffer_length)
			return;
	}
}

Sphere_Rasterizer::Sphere_Rasterizer(py::array& spheres, std::pair<int, int> output_resolution, int n_hf_entries, int max_buffer_length)
	: output_resolution(make_int2(std::get<0>(output_resolution), std::get<1>(output_resolution)))
	, n_hf_entries(n_hf_entries)
{
	allocate_spheres_cpu(spheres);
	if (n_spheres < max_buffer_length)
		buffer_length = n_spheres;
	else
		buffer_length = max_buffer_length;
	presort_spheres();
	spheres_gpu = allocate_spheres_on_gpu(spheres_cpu);

	extended_heightfield_gpu = allocate_float2_buffer_on_gpu(make_int3(std::get<0>(output_resolution), std::get<1>(output_resolution), buffer_length));
	normal_map_gpu = allocate_float3_buffer_on_gpu(make_int3(std::get<0>(output_resolution), std::get<1>(output_resolution), 1));
	z_buffer_gpu = allocate_float_buffer_on_gpu(make_int3(std::get<0>(output_resolution), std::get<1>(output_resolution), 1));
}

Sphere_Rasterizer::Sphere_Rasterizer(float2* extended_heightfield_gpu, py::array& spheres, std::pair<int, int> output_resolution, int n_hf_entries, int max_buffer_length) 
	: extended_heightfield_gpu(extended_heightfield_gpu)
	, output_resolution( make_int2(std::get<0>(output_resolution), std::get<1>(output_resolution) ) )
	, n_hf_entries(n_hf_entries)
{
	allocate_spheres_cpu(spheres);
	if (n_spheres < max_buffer_length)
		buffer_length = n_spheres;
	else
		buffer_length = max_buffer_length;
	presort_spheres();
	spheres_gpu = allocate_spheres_on_gpu(spheres_cpu);
	normal_map_gpu = allocate_float3_buffer_on_gpu(make_int3(std::get<0>(output_resolution), std::get<1>(output_resolution), 1));
	z_buffer_gpu = allocate_float_buffer_on_gpu(make_int3(std::get<0>(output_resolution), std::get<1>(output_resolution), 1));
}

Sphere_Rasterizer::Sphere_Rasterizer(float2* extended_heightfield_gpu, std::vector<Sphere>& spheres, std::pair<int, int> output_resolution, int n_hf_entries, int max_buffer_length)
	: extended_heightfield_gpu(extended_heightfield_gpu)
	, output_resolution(make_int2(std::get<0>(output_resolution), std::get<1>(output_resolution)))
	, n_hf_entries(n_hf_entries)
	, spheres_cpu(spheres)
{
	n_spheres = (int) spheres.size();
	if (n_spheres < max_buffer_length)
		buffer_length = n_spheres;
	else
		buffer_length = max_buffer_length;
	presort_spheres();
	spheres_gpu = allocate_spheres_on_gpu(spheres_cpu);
	normal_map_gpu = allocate_float3_buffer_on_gpu(make_int3(std::get<0>(output_resolution), std::get<1>(output_resolution), 1));
	z_buffer_gpu = allocate_float_buffer_on_gpu(make_int3(std::get<0>(output_resolution), std::get<1>(output_resolution), 1));
}

Sphere_Rasterizer::~Sphere_Rasterizer()
{
	// todo fix leaks
}

std::pair< py::array_t<float>, py::array_t<float> >  Sphere_Rasterizer::rasterize_spheres_py( float image_plane )
{
	rasterize_spheres( image_plane );
	return std::pair<py::array_t<float>, py::array_t<float> >(get_extended_height_field_py(), get_normal_map_py());
}

void Sphere_Rasterizer::rasterize_spheres( float image_plane )
{
	int2 grid_size = output_resolution;
	dim3 block_size(32, 32);
	dim3 num_blocks((grid_size.x + block_size.x - 1) / block_size.x, (grid_size.y + block_size.y - 1) / block_size.y);
	rasterize_sphere_kernel << <num_blocks, block_size >> > (spheres_gpu, spheres_cpu.size(), extended_heightfield_gpu, normal_map_gpu, z_buffer_gpu, output_resolution, buffer_length, n_hf_entries, image_plane, false );
	throw_on_cuda_error();
}

py::array_t<float> Sphere_Rasterizer::get_normal_map_py()
{
	auto normal_map_py = create_py_array(output_resolution.x, output_resolution.y, 3);
	cudaMemcpy(normal_map_py.request().ptr, normal_map_gpu, sizeof(float3) * output_resolution.x * output_resolution.y * 1, cudaMemcpyDeviceToHost);
	return normal_map_py;
}

std::vector<float> Sphere_Rasterizer::get_normal_map()
{
	std::vector<float> normal_map_cpu(3 * output_resolution.x * output_resolution.y );
	cudaMemcpy(&normal_map_cpu[0], normal_map_gpu, sizeof(float3) * output_resolution.x * output_resolution.y * 1, cudaMemcpyDeviceToHost);
	return normal_map_cpu;
}

py::array_t<float> Sphere_Rasterizer::get_extended_height_field_py()
{
	auto extended_hf_py = create_py_array(output_resolution.x, output_resolution.y, buffer_length * 2);
	cudaMemcpy(extended_hf_py.request().ptr, extended_heightfield_gpu, sizeof(float2) * output_resolution.x * output_resolution.y * buffer_length, cudaMemcpyDeviceToHost);
	return extended_hf_py;
}


void Sphere_Rasterizer::allocate_spheres_cpu(py::array& spheres)
{
	py::buffer_info info = spheres.request();
	if (info.ndim != 2)
		throw std::invalid_argument("spheres array is expected to be of dimensions nx4");
	if (info.shape[1] != 4)
		throw std::invalid_argument("spheres array is expected to be of dimensions nx4");
	if (info.format != "f")
		throw std::invalid_argument("spheres array is expected to be of dtype float32, found " + info.format);
	n_spheres = info.shape[0];
	spheres_cpu.resize(n_spheres);
	float* ptr = (float*) info.ptr;
	for (size_t i = 0; i < info.shape[0]; i++)
	{
		spheres_cpu[i].x = *(ptr++);
		spheres_cpu[i].y = *(ptr++);
		spheres_cpu[i].z = *(ptr++);
		spheres_cpu[i].r = *(ptr++);
	}
}

Sphere* Sphere_Rasterizer::allocate_spheres_on_gpu( const std::vector<Sphere>& spheres_cpu )
{
	Sphere* ptr_gpu;
	cudaMalloc((void**)&ptr_gpu, sizeof(Sphere) * n_spheres);
	cudaMemcpy(ptr_gpu, &spheres_cpu[0], sizeof(Sphere) * n_spheres, cudaMemcpyHostToDevice);
	return ptr_gpu;
}

void Sphere_Rasterizer::presort_spheres()
{
	struct {
		bool operator()(Sphere a, Sphere b) const { return a.z + a.r < b.z + b.r; }
	} bottomPosition;
	std::sort( spheres_cpu.begin(), spheres_cpu.end(), bottomPosition );
}
