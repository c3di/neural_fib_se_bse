#pragma once

#include "CSG_Resolver.h"

#include "cuda_utils.h"
#include "python_utils.h"

#include <stdexcept>

/* The CSG resolver sorts intervals (entry/exit) that were generated by the rasterizer, and solves the case
* that volumes overlap by merging the intervals
*/
__global__ void resolve_csg_kernel
(
	float2* extended_heightfield,
	int3 output_resolution,
	int n_hf_entries,
	bool debug)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx >= output_resolution.x)
		return;
	if (idy >= output_resolution.y)
		return;

	int pixel_index = idy * output_resolution.x + idx;

	float2& last_element_of_interest = extended_heightfield[pixel_index * output_resolution.z + n_hf_entries-1];

	int i = 1;
	while ( i < output_resolution.z && extended_heightfield[pixel_index * output_resolution.z + i].x != empty)
	{
		if (debug && idx == 74 && idy == 45)
		{
			printf("processing %i\n", i );
			if (debug && idx == 74 && idy == 45)
			{
				for (int ii = 0; ii < output_resolution.z; ii++)
					printf("  %i : %.2f-%.2f\n", ii, extended_heightfield[pixel_index * output_resolution.z + ii].x, extended_heightfield[pixel_index * output_resolution.z + ii].y);
			}
		}

		for (int j = i-1; j >= 0; j--)
		{
			float2& a = extended_heightfield[pixel_index * output_resolution.z + j];
			float2& b = extended_heightfield[pixel_index * output_resolution.z + j + 1];

			// early termination based on sorted input assumption
			if (b.x > last_element_of_interest.y)
				return;

			// case A: order ok
			if (a.y < b.x)
			{
			}

			// case B: swap
			else if (a.x > b.y)
			{
				float2 tmp = b;
				b = a;
				a = tmp;
			}

			// case C: merge
			else {
				float new_x = fminf(a.x, b.x);
				float new_y = fmaxf(a.y, b.y);
				a = make_float2(new_x, new_y);
				b = empty_interval;
				continue;
			} // merge
		}
		i++;
	}
	if (debug && idx == 74 && idy == 45)
	{
		for (int ii = 0; ii < output_resolution.z; ii++)
			printf("  %i : %.2f-%.2f\n", ii, extended_heightfield[pixel_index * output_resolution.z + ii].x, extended_heightfield[pixel_index * output_resolution.z + ii].y);
	}
}

CSG_Resolver::CSG_Resolver(float2* extended_heightfield_gpu, int3 buffer_size, int n_hf_entries)
	: extended_heightfield_gpu(extended_heightfield_gpu)
	, buffer_size(buffer_size)
	, n_hf_entries(n_hf_entries)
{
}

CSG_Resolver::CSG_Resolver(py::array_t<float> extended_heightfield, int n_hf_entries)
	: n_hf_entries(n_hf_entries) 
	, extended_heightfield_py(&extended_heightfield)
{
	extended_heightfield_cpu = get_extended_heightfield_cpu( extended_heightfield );
	extended_heightfield_gpu = allocate_buffer_on_gpu<float2>( buffer_size );
	cudaMemcpy(extended_heightfield_gpu, extended_heightfield_cpu, sizeof(float2) * buffer_size.x * buffer_size.y * buffer_size.z, cudaMemcpyHostToDevice);
}

CSG_Resolver::~CSG_Resolver()
{
}

py::array_t<float> CSG_Resolver::resolve_csg_py(float image_plane)
{
	resolve_csg(image_plane);
	cudaMemcpy(extended_heightfield_cpu, extended_heightfield_gpu, sizeof(float2) * buffer_size.x * buffer_size.y * buffer_size.z, cudaMemcpyDeviceToHost);
	return *extended_heightfield_py;
}

void CSG_Resolver::resolve_csg(float image_plane)
{
	dim3 block_size(32, 32);
	dim3 num_blocks((buffer_size.x + block_size.x - 1) / block_size.x, (buffer_size.y + block_size.y - 1) / block_size.y);

	resolve_csg_kernel << <num_blocks, block_size >> > (extended_heightfield_gpu, buffer_size, n_hf_entries, false);
	throw_on_cuda_error();
}

float* CSG_Resolver::get_extended_heightfield_cpu(py::array_t<float> extended_heightfield) 
{
	py::buffer_info info = extended_heightfield.request();
	if (info.ndim != 3)
		throw std::invalid_argument("extended_heightfield is expected to be of 3 dimensions");
	if (info.format != "f")
		throw std::invalid_argument("extended_heightfield is expected to be of dtype float32");
	buffer_size.x = info.shape[0];
	buffer_size.y = info.shape[1];
	buffer_size.z = info.shape[2] / 2;
	return (float*) info.ptr;
}