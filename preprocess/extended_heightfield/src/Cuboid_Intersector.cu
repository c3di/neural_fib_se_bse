#include "Cuboid_Intersector.h"

#include "cuda_utils.h"
#include "cuda_matrix.h"

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

/* The intersector computes the intersection points by brute force intersecting all primitives against a ray in positive z direction.
*  The operation is implemented by inverse transforming the ray, the intersecting the ray against a standard (z-up) cuboid.
*/
__global__ void intersect_cuboid_kernel(Cuboid* primitives,
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
	if (debug && idx == debug_position.x && idy == debug_position.y)
		printf("searching hit begin\n");

	int hit_index = 0;
	while (extended_heightfield[pixel_index * buffer_length + hit_index] != empty_interval)
	{
		if (debug && idx == debug_position.x && idy == debug_position.y)
		{
			float2 value = extended_heightfield[pixel_index * buffer_length + hit_index];
			printf("  hit index %i %.2f %.2f\n", hit_index, value.x, value.y);
		}
		hit_index++;
		if (hit_index > buffer_length)
			return;
	}

	if (debug && idx == debug_position.x && idy == debug_position.y)
		printf("starting cuboid insertion at hit index %i\n", hit_index);

	// loop over all spheres
	for (int primitive_id = 0; primitive_id < n_primitives; primitive_id++)
	{
		if (debug && idx == debug_position.x && idy == debug_position.y)
			printf("  cuboid ID %i\n", primitive_id);

		const Cuboid& cuboid = primitives[primitive_id];

		if (debug && idx == debug_position.x && idy == debug_position.y)
			printf("  aabb test size %.2f %.2f %.2f aabb %.2f %.2f %.2f\n", cuboid.size.x, cuboid.size.y, cuboid.size.z, cuboid.aabb.x, cuboid.aabb.y, cuboid.aabb.z );

		if ((pixel_x < cuboid.position.x - cuboid.aabb.x) || (pixel_x > cuboid.position.x + cuboid.aabb.x)
		 || (pixel_y < cuboid.position.y - cuboid.aabb.y) || (pixel_y > cuboid.position.y + cuboid.aabb.y)
		 || (image_plane_z > cuboid.position.z + cuboid.aabb.z))
			continue;

		if (debug && idx == debug_position.x && idy == debug_position.y)
			printf("  aabb test passed\n");

		float3 ray_origin    = make_float3(pixel_x- cuboid.position.x, pixel_y - cuboid.position.y, image_plane_z - cuboid.position.z);
		float3 ray_direction = make_float3(0.0f,                        0.0f,                          1.0f);

		if ( debug && idx == debug_position.x && idy == debug_position.y )
		{
			printf("pixel               %.2f %.2f\n", pixel_x, pixel_y);
			printf("cylinder            %.2f %.2f %.2f dir %.2f %.2f %.2f %.2f\n", cuboid.position.x, cuboid.position.y, cuboid.position.z, cuboid.orientation.x, cuboid.orientation.y, cuboid.orientation.z, cuboid.orientation.w );
			printf("original ray origin %.2f %.2f %.2f direction %.2f %.2f %.2f \n", ray_origin.x, ray_origin.y, ray_origin.z, ray_direction.x, ray_direction.y, ray_direction.z);
		}

		// Matrix3x3 object_to_world = getFromEulerAngles( cylinder.orientation );
		const float4& object_to_world = cuboid.orientation;
		if ( debug && idx == debug_position.x && idy == debug_position.y )
		{
			printf("object_to_world\n");
			printf("  %.2f %.2f %.2f %.2f\n", object_to_world.x, object_to_world.y, object_to_world.z, object_to_world.w);
			printf("\n");
		}

		float4 world_to_object = getInverseQuaternion(object_to_world);

		if ( debug && idx == debug_position.x && idy == debug_position.y )
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

		if ( debug && idx == debug_position.x && idy == debug_position.y )
			printf("transformed  ray origin %.2f %.2f %.2f direction %.2f %.2f %.2f \n", ray_origin.x, ray_origin.y, ray_origin.z, ray_direction.x, ray_direction.y, ray_direction.z);

		float3 normal = make_float3(1.0f, 0.0f, 0.0f);

		float tmin = (-cuboid.size.x - ray_origin.x) / ray_direction.x;
		float tmax = ( cuboid.size.x - ray_origin.x) / ray_direction.x;

		if (tmin > tmax) 
			swap(tmin, tmax);

		if ( debug && idx == debug_position.x && idy == debug_position.y )
			printf("intersect with x plane %.2f %.2f \n", tmin, tmax);

		float tymin = (-cuboid.size.y - ray_origin.y) / ray_direction.y;
		float tymax = ( cuboid.size.y - ray_origin.y) / ray_direction.y;

		if (tymin > tymax) 
			swap(tymin, tymax);

		if ( debug && idx == debug_position.x && idy == debug_position.y )
			printf("intersect with y plane %.2f %.2f \n", tymin, tymax);

		if ((tmin > tymax) || (tymin > tmax))
		{
			if ( debug && idx == debug_position.x && idy == debug_position.y )
				printf("no intersect because of tmin> tymax or tymin > tmax\n");
			continue;
		}

		if (tymin > tmin)
		{
			tmin = tymin;
			normal = make_float3(0.0f, 1.0f, 0.0f);;
		}

		if (tymax < tmax)
			tmax = tymax;

		if (debug && idx == debug_position.x && idy == debug_position.y)
			printf("intersect considering y plane %.2f %.2f \n", tmin, tmax);

		float tzmin = (-cuboid.size.z - ray_origin.z) / ray_direction.z;
		float tzmax = ( cuboid.size.z - ray_origin.z) / ray_direction.z;

		if (debug && idx == debug_position.x && idy == debug_position.y)
			printf("intersect with z plane %.2f - %.2f / %.2f = %.2f\n", cuboid.size.z, ray_origin.z, ray_direction.z, tzmin);


		if (tzmin > tzmax) 
			swap(tzmin, tzmax);

		if (debug && idx == debug_position.x && idy == debug_position.y)
			printf("intersect with z plane %.2f %.2f \n", tzmin, tzmax);

		if ((tmin > tzmax) || (tzmin > tmax))
			continue;

		if (tzmin > tmin)
		{
			tmin = tzmin;
			normal = make_float3(0.0f, 0.0f, 1.0f);
		}

		if (tzmax < tmax)
			tmax = tzmax;

		if (debug && idx == debug_position.x && idy == debug_position.y)
			printf("final intersect at %.2f %.2f \n", tmin, tmax);

		bool cut_case = false;
		// handle the case that the sphere is cut by the image place 
		if (tmin < 0.0f)
		{
			tmin = 0.0f;
			cut_case = true;
		}

		extended_heightfield[pixel_index * buffer_length + hit_index] = make_float2( tmin, tmax );
		hit_index++;

		// write the normal map
		if (tmin < z_buffer[pixel_index])
		{
			z_buffer[pixel_index] = tmin;
			if (cut_case)
			{
				normal_map[pixel_index] = make_float3(0.0f, 0.0f, 1.0f);
			}
			else
			{
				normal = getPointTransformedByQuaternion(object_to_world, normal);
				normal = getNormalizedVec(normal);
				if (normal.z < 0.0f)
					normal = -1.0f * normal;
				normal_map[pixel_index] = normal;
			}
		}

		if (hit_index > buffer_length)
			return;
	}
}

Cuboid_Intersector::Cuboid_Intersector(std::tuple<int, int> output_resolution, int n_hf_entries, int max_buffer_length)
	: Abstract_Intersector<Cuboid>(output_resolution, n_hf_entries, max_buffer_length )
{
}

Cuboid_Intersector::Cuboid_Intersector(float2* extended_heightfield_gpu, float* z_buffer_gpu, float3* normal_map_gpu, std::tuple<int, int> output_resolution, int n_hf_entries, int max_buffer_length)
	: Abstract_Intersector<Cuboid>(extended_heightfield_gpu, z_buffer_gpu, normal_map_gpu, output_resolution, n_hf_entries, max_buffer_length)
{
}

Cuboid_Intersector::~Cuboid_Intersector()
{
}

void Cuboid_Intersector::intersect( float image_plane )
{
	int2 grid_size = output_resolution;
	dim3 block_size(16, 16);
	dim3 num_blocks((grid_size.x + block_size.x - 1) / block_size.x, (grid_size.y + block_size.y - 1) / block_size.y);
	intersect_cuboid_kernel << <num_blocks, block_size >> > (primitives_gpu, primitives_cpu.size(), extended_heightfield->gpu_ptr(), normal_map->gpu_ptr(), z_buffer->gpu_ptr(), output_resolution, buffer_length, n_hf_entries, image_plane, false, make_int2(425, 425) );
	throw_on_cuda_error();
}

void Cuboid_Intersector::assign_aabb()
{
	for (Cuboid& cuboid : primitives_cpu)
	{
		float3 corners[8];
		corners[0] = make_float3(cuboid.size.x, cuboid.size.y, cuboid.size.z);
		corners[1] = make_float3(cuboid.size.x, cuboid.size.y, -cuboid.size.z);
		corners[2] = make_float3(cuboid.size.x, -cuboid.size.y, cuboid.size.z);
		corners[3] = make_float3(cuboid.size.x, -cuboid.size.y, -cuboid.size.z);
		corners[4] = make_float3(-cuboid.size.x, cuboid.size.y, cuboid.size.z);
		corners[5] = make_float3(-cuboid.size.x, cuboid.size.y, -cuboid.size.z);
		corners[6] = make_float3(-cuboid.size.x, -cuboid.size.y, cuboid.size.z);
		corners[7] = make_float3(-cuboid.size.x, -cuboid.size.y, -cuboid.size.z);

		for (int i = 0; i < 8; i++)
			corners[i] = getPointTransformedByQuaternion(cuboid.orientation, corners[i]);

		cuboid.aabb = make_float3( 0.0f, 0.0f, 0.0f );

		for (int i = 0; i < 8; i++)
		{
			cuboid.aabb.x = fmaxf(cuboid.aabb.x, fabsf(corners[i].x));
			cuboid.aabb.y = fmaxf(cuboid.aabb.y, fabsf(corners[i].y));
			cuboid.aabb.z = fmaxf(cuboid.aabb.z, fabsf(corners[i].z));
		}
	}
}
