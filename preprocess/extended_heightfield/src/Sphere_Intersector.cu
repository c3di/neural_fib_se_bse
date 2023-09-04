#include "Sphere_Intersector.h"

#include "cuda_utils.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>

/* The intersector computes the intersection points by brute force intersecting all spheres against a ray in positive z direction
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

	int pixel_index = idy * output_resolution.x + idx;

	// initialize z_buffer
	z_buffer[pixel_index] = empty;

	const float pixel_x = (float) idx;
	const float pixel_y = (float) idy;

	// search beginning
	int hit_index = 0;
	while (extended_heightfield[pixel_index * buffer_length + hit_index] != empty_interval)
		hit_index++;

	// loop over all spheres
	for (int sphere_id = 0; sphere_id < n_spheres; sphere_id++)
	{
		const Sphere& sphere = spheres[sphere_id];

		const float dz = fabsf( sphere.position.z - image_plane_z);

		// early termination if sphere behind image plane
		if ( dz <= -sphere.r )
			continue;

		if (debug && idx == 74 && idy == 45)
			printf("    : front of image plance\n");

		// calculate entry and exit point by computing both solutions to r^2 = (x-x0)^2 + (y-y0)^2 + (z-z0)^2
		const float dx = pixel_x - sphere.position.x;
		const float dy = pixel_y - sphere.position.y;

		// check if intersection point exists
		if (dx * dx + dy * dy > sphere.r * sphere.r)
			continue;

		const float square_term = sqrtf( sphere.r * sphere.r - dx * dx - dy * dy );
		float entry = sphere.position.z - square_term;
		float exit  = sphere.position.z + square_term;

		bool cut_case = false;
		// handle the case that the sphere is cut by the image place 
		if (entry < image_plane_z)
		{
			entry = image_plane_z;
			cut_case = true;
		}

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

Sphere_Intersector::Sphere_Intersector(std::tuple<int, int> output_resolution, int n_hf_entries, int max_buffer_length)
	: Abstract_Intersector<Sphere>(output_resolution, n_hf_entries, max_buffer_length )
{
}

Sphere_Intersector::Sphere_Intersector(float2* extended_heightfield_gpu, float* z_buffer_gpu, float3* normal_map_gpu, std::tuple<int, int> output_resolution, int n_hf_entries, int max_buffer_length)
	: Abstract_Intersector<Sphere>(extended_heightfield_gpu, z_buffer_gpu, normal_map_gpu, output_resolution, n_hf_entries, max_buffer_length)
{
}

Sphere_Intersector::~Sphere_Intersector()
{
}

void Sphere_Intersector::intersect( float image_plane )
{
	int2 grid_size = output_resolution;
	dim3 block_size(32, 32);
	dim3 num_blocks((grid_size.x + block_size.x - 1) / block_size.x, (grid_size.y + block_size.y - 1) / block_size.y);
	rasterize_sphere_kernel << <num_blocks, block_size >> > (primitives_gpu, primitives_cpu.size(), extended_heightfield->gpu_ptr(), normal_map->gpu_ptr(), z_buffer->gpu_ptr(), output_resolution, buffer_length, n_hf_entries, image_plane, false );
	throw_on_cuda_error();
}

void Sphere_Intersector::assign_aabb()
{
}
