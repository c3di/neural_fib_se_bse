#pragma once

#include "Cylinder_Intersector.h"

#include "cuda_utils.h"
#include "cuda_matrix.h"
#include "math_functions.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>

#include "cuda_matrix.h"

__device__ inline void swap(float& a, float& b) 
{
	const float tmp = a;
	a = b;
	b = tmp;
}

__device__ inline bool quadratic(float a, float b, float c, float& t0, float& t1)
{
	float discrim = b * b - 4.0f * a * c;
	if (discrim <= 0.f)
		return false;
	float rootDiscrim = sqrtf(discrim);
	float q;
	if (b < 0.0f) 
		q = -0.5f * (b - rootDiscrim);
	else
		q = -0.5f * (b + rootDiscrim);
	t0 = q / a;
	t1 = c / q;
	if (t0 > t1)
		swap(t0, t1);
	return true;
}

/* The intersector computes the intersection points by brute force intersecting all cylinders against a ray in positive z direction.
*  The operation is implemented by inverse transforming the ray, the intersecting the ray against a standard (z-up) cylinder.
*/
__global__ void rasterize_cylinder_kernel(Cylinder* primitives,
								        int n_primitives,
								        float2* extended_heightfield, // contains entry/exit information as float2 per pixel
									    float3* normal_map,
									    float* z_buffer,
	                                    int2 output_resolution,
										int buffer_length,
										int n_hf_entries,
										float image_plane_z,
										bool debug, 
										int2 debug_position )
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

	if (debug && idx == debug_position.x && idy == debug_position.y)
		printf("found first free entry in heightfield buffer at index %i\n", hit_index);

	// loop over all spheres
	for (int primitive_id = 0; primitive_id < n_primitives; primitive_id++)
	{
		const Cylinder& cylinder = primitives[primitive_id];

		if ((pixel_x < cylinder.position.x - cylinder.aabb.x) || (pixel_x > cylinder.position.x + cylinder.aabb.x)
		 || (pixel_y < cylinder.position.y - cylinder.aabb.y) || (pixel_y > cylinder.position.y + cylinder.aabb.y)
		 || (image_plane_z > cylinder.position.z + cylinder.aabb.z))
			continue;

		float3 ray_origin    = make_float3(pixel_x-cylinder.position.x, pixel_y - cylinder.position.y, image_plane_z - cylinder.position.z);
		float3 ray_direction = make_float3(0.0f,                        0.0f,                          1.0f);

		if (debug && idx == 74 && idy == 45) 
		{
			printf("pixel               %.2f %.2f\n", pixel_x, pixel_y);
			printf("cylinder            %.2f %.2f %.2f dir %.2f %.2f %.2f %.2f\n", cylinder.position.x, cylinder.position.y, cylinder.position.z, cylinder.orientation.x, cylinder.orientation.y, cylinder.orientation.z, cylinder.orientation.w );
			printf("original ray origin %.2f %.2f %.2f direction %.2f %.2f %.2f \n", ray_origin.x, ray_origin.y, ray_origin.z, ray_direction.x, ray_direction.y, ray_direction.z);
		}

		// Matrix3x3 object_to_world = getFromEulerAngles( cylinder.orientation );
		const float4& object_to_world = cylinder.orientation;
		if (debug && idx == debug_position.x && idy == debug_position.y)
		{
			printf("object_to_world\n");
			printf("  %.2f %.2f %.2f %.2f\n", object_to_world.x, object_to_world.y, object_to_world.z, object_to_world.w);
			printf("\n");
		}

		float4 world_to_object = getInverseQuaternion(object_to_world);

		if (debug && idx == debug_position.x && idy == debug_position.y)
		{
			printf("world_to_object\n");
			printf("  %.2f %.2f %.2f %.2f\n", world_to_object.x, world_to_object.y, world_to_object.z, world_to_object.w);
			printf("\n");
		}

		ray_origin    = getPointTransformedByQuaternion(world_to_object, ray_origin);
		ray_direction = getPointTransformedByQuaternion(world_to_object, ray_direction);
		if (debug && idx == debug_position.x && idy == debug_position.y)
			printf("unnormalized ray origin %.2f %.2f %.2f direction %.2f %.2f %.2f \n", ray_origin.x, ray_origin.y, ray_origin.z, ray_direction.x, ray_direction.y, ray_direction.z);
		ray_direction = getNormalizedVec(ray_direction);

		if (debug && idx == debug_position.x && idy == debug_position.y)
			printf("transformed ray origin %.2f %.2f %.2f direction %.2f %.2f %.2f \n", ray_origin.x, ray_origin.y, ray_origin.z, ray_direction.x, ray_direction.y, ray_direction.z);

		float a = ray_direction.x * ray_direction.x + ray_direction.y * ray_direction.y;
		float b = 2.0f * (ray_direction.x * ray_origin.x + ray_direction.y * ray_origin.y);
		float c = ray_origin.x * ray_origin.x + ray_origin.y * ray_origin.y - cylinder.r * cylinder.r;

		float cap_t0 = (-cylinder.l - ray_origin.z) / ray_direction.z;
		float cap_t1 = ( cylinder.l - ray_origin.z) / ray_direction.z;

		if (cap_t0 > cap_t1)
			swap(cap_t0, cap_t1);

		const float3 cap_entry_point = ray_origin + cap_t0 * ray_direction;
		const float3 cap_exit_point  = ray_origin + cap_t1 * ray_direction;

		if (debug && idx == debug_position.x && idy == debug_position.y) 
		{
			printf("cap_entry_point %.2f %.2f %.2f / %.2f %.2f %.2f \n", cap_entry_point.x, cap_entry_point.y, cap_entry_point.z, cap_exit_point.x, cap_exit_point.y, cap_exit_point.z);
			printf("ray origin      %.2f %.2f %.2f direction %.2f %.2f %.2f \n", ray_origin.x, ray_origin.y, ray_origin.z, ray_direction.x, ray_direction.y, ray_direction.z);
		}

		bool cap_hit0 = (cap_entry_point.x * cap_entry_point.x + cap_entry_point.y * cap_entry_point.y <= cylinder.r * cylinder.r);
		bool cap_hit1 = (cap_exit_point.x  * cap_exit_point.x  + cap_exit_point.y  * cap_exit_point.y  <= cylinder.r * cylinder.r);

		bool side_hit0 = false;
		bool side_hit1 = false;

		float3 normal;

		float side_t0, side_t1;
		if (quadratic(a, b, c, side_t0, side_t1))
		{
			const float3 side_entry_point = ray_origin + side_t0 * ray_direction;
			const float3 side_exit_point  = ray_origin + side_t1 * ray_direction;

			if ((side_entry_point.z >= -cylinder.l) && (side_entry_point.z <= cylinder.l))
			{
				side_hit0 = true;
				normal = make_float3(side_entry_point.x / cylinder.r, side_entry_point.y / cylinder.r, 0.0f);
			}
			if ((side_exit_point.z >= -cylinder.l) && (side_exit_point.z <= cylinder.l))
				side_hit1 = true;
		}

		if (!cap_hit0 && !side_hit0 && !cap_hit1 && !side_hit1)
			continue;

		float t0, t1;
		if (cap_hit0)
		{
			t0 = cap_t0;
			normal = make_float3( 0.0f, 0.0f, 1.0f );
		}
		else 
			t0 = side_t0;

		if (cap_hit1)
			t1 = cap_t1;
		else
			t1 = side_t1;

		bool cut_case = false;
		// handle the case that the sphere is cut by the image place 
		if (t0 < 0.0f)
		{
			t0 = 0.0f;
			cut_case = true;
		}

		if (debug && idx == debug_position.x && idy == debug_position.y) {
			printf("found intersection cylinder center %.2f %.2f %.2f orientation %.2f %.2f %.2f %.2f radius %.2f length %.2f\n", cylinder.position.x, cylinder.position.y, cylinder.position.z, cylinder.orientation.x, cylinder.orientation.y, cylinder.orientation.z, cylinder.orientation.w, cylinder.r, cylinder.l);
			printf("      intersection state           cap %i %i side %i %i t0 t1 %.2f %.2f\n", cap_hit0, cap_hit1, side_hit0, side_hit1, t0, t1);
		}

		extended_heightfield[pixel_index * buffer_length + hit_index] = make_float2( t0, t1 );
		hit_index++;

		// write the normal map
		if (t0 < z_buffer[pixel_index])
		{
			z_buffer[pixel_index] = t0;
			if (cut_case)
			{
				normal_map[pixel_index] = make_float3(0.0f, 0.0f, 1.0f);
			}
			else
			{
				if (debug && idx == debug_position.x && idy == debug_position.y) {
					printf("      normal object space %.2f %.2f %.2f\n", normal.x, normal.y, normal.z);
				}
				normal = getPointTransformedByQuaternion(object_to_world, normal);
				normal = getNormalizedVec(normal);
				if (debug && idx == debug_position.x && idy == debug_position.y) {
					printf("      normal transfomed   %.2f %.2f %.2f\n", normal.x, normal.y, normal.z);
				}
				if (normal.z < 0.0f)
					normal = -1.0f * normal;
				normal_map[pixel_index] = normal;
			}
		}

		if (hit_index > buffer_length)
			return;
	}
	if (debug && idx == debug_position.x && idy == debug_position.y) {
		printf("final normal %.2f %.2f %.2f\n", normal_map[pixel_index].x, normal_map[pixel_index].y, normal_map[pixel_index].z);
	}
}

Cylinder_Intersector::Cylinder_Intersector(std::tuple<int, int> output_resolution, int n_hf_entries, int max_buffer_length)
	: Abstract_Intersector<Cylinder>(output_resolution, n_hf_entries, max_buffer_length )
{
}

Cylinder_Intersector::Cylinder_Intersector(float2* extended_heightfield_gpu, float* z_buffer_gpu, float3* normal_map_gpu, std::tuple<int, int> output_resolution, int n_hf_entries, int max_buffer_length)
	: Abstract_Intersector<Cylinder>(extended_heightfield_gpu, z_buffer_gpu, normal_map_gpu, output_resolution, n_hf_entries, max_buffer_length)
{
}

Cylinder_Intersector::~Cylinder_Intersector()
{
}

void Cylinder_Intersector::intersect( float image_plane )
{
	int2 grid_size = output_resolution;
	dim3 block_size(16, 16);
	dim3 num_blocks((grid_size.x + block_size.x - 1) / block_size.x, (grid_size.y + block_size.y - 1) / block_size.y);
	rasterize_cylinder_kernel << <num_blocks, block_size >> > (primitives_gpu, primitives_cpu.size(), extended_heightfield->gpu_ptr(), normal_map->gpu_ptr(), z_buffer->gpu_ptr(), output_resolution, buffer_length, n_hf_entries, image_plane, true, make_int2(425, 425) );
	throw_on_cuda_error();
}

// https://iquilezles.org/articles/diskbbox/
void Cylinder_Intersector::assign_aabb()
{
	for (Cylinder& cylinder : primitives_cpu)
	{
		const float3 a = getPointTransformedByQuaternion( cylinder.orientation, make_float3(0.0, 0.0, 1.0) );
		const float a_dot_a = dot( a, a);
		const float3 e = cylinder.r * sqrtf(1.0f - a * a / a_dot_a );
		cylinder.aabb = abs(e);
	}
}
