#include "HeightFieldExtractor.h"
#include "Sphere_Intersector.h"
#include "Cylinder_Intersector.h"
#include "Cuboid_Intersector.h"
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

	int pixel_index = idy * output_resolution.x + idx;
	for (int i = 0; i < n_hf_entries; i++)
	{
		result[pixel_index * n_hf_entries + i] = extended_heightfield[pixel_index * output_resolution.z + i];
	}
}

HeightFieldExtractor::HeightFieldExtractor( std::tuple<int, int> output_resolution, int n_hf_entries, int max_buffer_length )
	: output_resolution( make_int2(std::get<0>(output_resolution), std::get<1>(output_resolution)) )
	, n_hf_entries(n_hf_entries)
	, max_buffer_length(max_buffer_length)
{
	int3 extended_heightfield_size = make_int3(std::get<0>(output_resolution), std::get<1>(output_resolution), max_buffer_length );
	extended_heightfield_gpu = allocate_buffer_on_gpu<float2>(extended_heightfield_size, empty_interval);
	result_gpu = allocate_buffer_on_gpu<float2>(make_int3(std::get<0>(output_resolution), std::get<1>(output_resolution), n_hf_entries));
	cudaMallocHost(&result_cpu, sizeof(float2) * extended_heightfield_size.x * extended_heightfield_size.y * n_hf_entries);
	csg_resolver = new CSG_Resolver(extended_heightfield_gpu, make_int3(std::get<0>(output_resolution), std::get<1>(output_resolution), max_buffer_length), n_hf_entries );
	z_buffer_gpu = allocate_buffer_on_gpu<float>(make_int3(std::get<0>(output_resolution), std::get<1>(output_resolution), 1));
	normal_map_gpu = allocate_buffer_on_gpu<float3>(make_int3(std::get<0>(output_resolution), std::get<1>(output_resolution), 1));
}

HeightFieldExtractor::~HeightFieldExtractor()
{
	cudaFree(normal_map_gpu);
	cudaFree(z_buffer_gpu);
	delete csg_resolver;
	cudaFree(result_gpu);
	cudaFree(extended_heightfield_gpu);
	cudaFreeHost(result_cpu);
}

void HeightFieldExtractor::add_spheres_py(py::array& spheres)
{
	auto method = new Sphere_Intersector(extended_heightfield_gpu, z_buffer_gpu, normal_map_gpu, as_tuple(output_resolution), n_hf_entries, max_buffer_length);
	method->add_primitives_py(spheres);
	intersectors.push_back(method);
}

void HeightFieldExtractor::add_spheres(std::vector<Sphere>& spheres)
{
	auto method = new Sphere_Intersector(extended_heightfield_gpu, z_buffer_gpu, normal_map_gpu, as_tuple(output_resolution), n_hf_entries, max_buffer_length);
	method->add_primitives(spheres);
	intersectors.push_back(method);
}

void HeightFieldExtractor::add_cylinders_py(py::array& cylinders)
{
	auto method = new Cylinder_Intersector(extended_heightfield_gpu, z_buffer_gpu, normal_map_gpu, as_tuple(output_resolution), n_hf_entries, max_buffer_length);
	method->add_primitives_py(cylinders);
	intersectors.push_back(method);
}

void HeightFieldExtractor::add_cylinders(std::vector<Cylinder>& cylinders)
{
	auto method = new Cylinder_Intersector(extended_heightfield_gpu, z_buffer_gpu, normal_map_gpu, as_tuple(output_resolution), n_hf_entries, max_buffer_length);
	method->add_primitives(cylinders);
	intersectors.push_back(method);
 }

void HeightFieldExtractor::add_cuboids_py(py::array& cuboids)
{
	auto method = new Cuboid_Intersector(extended_heightfield_gpu, z_buffer_gpu, normal_map_gpu, as_tuple(output_resolution), n_hf_entries, max_buffer_length);
	method->add_primitives_py(cuboids);
	intersectors.push_back(method);
}

void HeightFieldExtractor::add_cuboids(std::vector<Cuboid>& cuboids)
{
	auto method = new Cuboid_Intersector(extended_heightfield_gpu, z_buffer_gpu, normal_map_gpu, as_tuple(output_resolution), n_hf_entries, max_buffer_length);
	method->add_primitives(cuboids);
	intersectors.push_back(method);
}

std::tuple<float2*, float3*> HeightFieldExtractor::extract_data_representation(float image_plane)
{
	intersect( image_plane );
	return std::tuple<float2*, float3*>(collect_extended_heightfield(), intersectors[0]->get_normal_map());
}

std::tuple< py::array_t<float2>, py::array_t<float3>>  HeightFieldExtractor::extract_data_representation_py(float image_plane)
{
	intersect( image_plane );
	return std::tuple< py::array_t<float2>, py::array_t<float3>>( collect_extended_heightfield_py(), intersectors[0]->get_normal_map_py() );
}

void HeightFieldExtractor::intersect(float image_plane)
{
	for (auto intersectors : intersectors) {
		intersectors->intersect(image_plane);
		cudaDeviceSynchronize();
	}
	csg_resolver->resolve_csg(image_plane);
	cudaDeviceSynchronize();
}

py::array_t<float2> HeightFieldExtractor::collect_extended_heightfield_py()
{
	call_result_collection_kernel();
	cudaDeviceSynchronize();
	auto pyarray = create_py_array(output_resolution.x, output_resolution.y, n_hf_entries * 2);
	cudaMemcpy(pyarray.request().ptr, result_gpu, sizeof(float2) * output_resolution.x * output_resolution.y * n_hf_entries, cudaMemcpyDeviceToHost);
	return pyarray;
}

float2* HeightFieldExtractor::collect_extended_heightfield()
{
	call_result_collection_kernel();
	cudaMemcpy( result_cpu, result_gpu, sizeof(float2) * output_resolution.x * output_resolution.y * n_hf_entries, cudaMemcpyDeviceToHost);
	return result_cpu;
}	

void HeightFieldExtractor::call_result_collection_kernel()
{
	dim3 block_size(32, 32);
	dim3 num_blocks((output_resolution.x + block_size.x - 1) / block_size.x, (output_resolution.y + block_size.y - 1) / block_size.y);
	int3 buffer_size = make_int3(output_resolution.x, output_resolution.y, max_buffer_length);
	collect_result_kernel << <num_blocks, block_size >> > (extended_heightfield_gpu, result_gpu, buffer_size, n_hf_entries);
}
