#pragma once

#include "HeightFieldExtractor.h"
#include "Sphere_Rasterizer.h"
#include "Cylinder_Rasterizer.h"
#include "CSG_Resolver.h"

#include "cuda_utils.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>

__global__ void collect_result_kernel( float2* extended_heightfield, float2* result, int3 output_resolution, int n_hf_entries )
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx >= output_resolution.x)
		return;
	if (idy >= output_resolution.y)
		return;

	int pixel_index = idx * output_resolution.y + idy;
	for (int i = 0; i < n_hf_entries; i++)
	{
		result[pixel_index * n_hf_entries + i] = extended_heightfield[pixel_index * output_resolution.z + i];
	}
}

HeightFieldExtractor::HeightFieldExtractor( std::pair<int, int> output_resolution, int n_hf_entries, int max_buffer_length )
	: output_resolution( make_int2(std::get<0>(output_resolution), std::get<1>(output_resolution)) )
	, n_hf_entries(n_hf_entries)
	, max_buffer_length(max_buffer_length)
{
	extended_heightfield_gpu = allocate_float2_buffer_on_gpu(make_int3(std::get<0>(output_resolution), std::get<1>(output_resolution), max_buffer_length));
	result_gpu = allocate_float2_buffer_on_gpu(make_int3(std::get<0>(output_resolution), std::get<1>(output_resolution), n_hf_entries));
	csg_resolver = new CSG_Resolver(extended_heightfield_gpu, make_int3(std::get<0>(output_resolution), std::get<1>(output_resolution), max_buffer_length), n_hf_entries );
	z_buffer_gpu = allocate_float_buffer_on_gpu(make_int3(std::get<0>(output_resolution), std::get<1>(output_resolution), 1));
	normal_map_gpu = allocate_float3_buffer_on_gpu(make_int3(std::get<0>(output_resolution), std::get<1>(output_resolution), 1));
}

HeightFieldExtractor::~HeightFieldExtractor()
{
	delete csg_resolver;
	delete z_buffer_gpu;
	delete normal_map_gpu;
	// delete sphere_rasterizer;
	// delete result_gpu;
}

void HeightFieldExtractor::add_spheres_py(py::array& spheres)
{
	auto method = new Sphere_Rasterizer(extended_heightfield_gpu, z_buffer_gpu, normal_map_gpu, tuple(output_resolution), n_hf_entries, max_buffer_length);
	method->add_primitives_py(spheres);
	rasterizer.push_back(method);
}

void HeightFieldExtractor::add_spheres(std::vector<Sphere>& spheres)
{
	auto method = new Sphere_Rasterizer(extended_heightfield_gpu, z_buffer_gpu, normal_map_gpu, tuple(output_resolution), n_hf_entries, max_buffer_length);
	method->add_primitives(spheres);
	rasterizer.push_back(method);
}

void HeightFieldExtractor::add_cylinders_py(py::array& cylinders)
{
	auto method = new Cylinder_Rasterizer(extended_heightfield_gpu, z_buffer_gpu, normal_map_gpu, tuple(output_resolution), n_hf_entries, max_buffer_length);
	method->add_primitives_py(cylinders);
	rasterizer.push_back(method);
}

void HeightFieldExtractor::add_cylinders(std::vector<Cylinder>& cylinders)
{
	auto method = new Cylinder_Rasterizer(extended_heightfield_gpu, z_buffer_gpu, normal_map_gpu, tuple(output_resolution), n_hf_entries, max_buffer_length);
	method->add_primitives(cylinders);
	rasterizer.push_back(method);
}

std::pair<std::vector<float>, std::vector<float>> HeightFieldExtractor::extract_data_representation(float image_plane)
{
	rasterize(image_plane);
	return std::pair<std::vector<float>, std::vector<float>>(collect_extended_heightfield(), rasterizer[0]->get_normal_map());
}

std::pair< py::array_t<float>, py::array_t<float>>  HeightFieldExtractor::extract_data_representation_py(float image_plane)
{
	rasterize( image_plane );
	return std::pair< py::array_t<float>, py::array_t<float>>( collect_extended_heightfield_py(), rasterizer[0]->get_normal_map_py() );
}

void HeightFieldExtractor::rasterize(float image_plane)
{
	for ( auto rasterizer : rasterizer )
		rasterizer->rasterize( image_plane );
	csg_resolver->resolve_csg(image_plane);
}

py::array_t<float> HeightFieldExtractor::collect_extended_heightfield_py()
{
	call_result_collection_kernel();
	auto pyarray = create_py_array(output_resolution.x, output_resolution.y, n_hf_entries * 2);
	cudaMemcpy(pyarray.request().ptr, result_gpu, sizeof(float2) * output_resolution.x * output_resolution.y * n_hf_entries, cudaMemcpyDeviceToHost);
	return pyarray;
}

std::vector<float> HeightFieldExtractor::collect_extended_heightfield()
{
	call_result_collection_kernel();
	std::vector<float> result_cpu(2 * output_resolution.x * output_resolution.y * n_hf_entries);
	cudaMemcpy(&result_cpu[0], result_gpu, sizeof(float2) * output_resolution.x * output_resolution.y * n_hf_entries, cudaMemcpyDeviceToHost);
	return result_cpu;
}	

void HeightFieldExtractor::call_result_collection_kernel()
{
	dim3 block_size(32, 32);
	dim3 num_blocks((output_resolution.x + block_size.x - 1) / block_size.x, (output_resolution.y + block_size.y - 1) / block_size.y);
	int3 buffer_size = make_int3(output_resolution.x, output_resolution.y, max_buffer_length);
	collect_result_kernel << <num_blocks, block_size >> > (extended_heightfield_gpu, result_gpu, buffer_size, n_hf_entries);
}
