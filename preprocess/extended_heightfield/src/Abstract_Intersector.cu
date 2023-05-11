#pragma once

#include "Abstract_Intersector.h"

#include "cuda_utils.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <string>

template<class Primitive>
Abstract_Intersector<Primitive>::Abstract_Intersector<Primitive>(std::pair<int, int> output_resolution, int n_hf_entries, int buffer_length)
	: output_resolution(make_int2(std::get<0>(output_resolution), std::get<1>(output_resolution)))
	, n_hf_entries(n_hf_entries)
	, buffer_length(buffer_length)
{
	extended_heightfield_gpu = allocate_buffer_on_gpu<float2>(make_int3(std::get<0>(output_resolution), std::get<1>(output_resolution), buffer_length), empty_interval);
	normal_map_gpu = allocate_buffer_on_gpu<float3>(make_int3(std::get<0>(output_resolution), std::get<1>(output_resolution), 1));
	z_buffer_gpu = allocate_buffer_on_gpu<float>(make_int3(std::get<0>(output_resolution), std::get<1>(output_resolution), 1));
}

template<class Primitive>
Abstract_Intersector<Primitive>::Abstract_Intersector<Primitive>(float2* extended_heightfield_gpu, float* z_buffer_gpu, float3* normal_map_gpu, std::pair<int, int> output_resolution, int n_hf_entries, int buffer_length)
	: extended_heightfield_gpu(extended_heightfield_gpu)
	, z_buffer_gpu(z_buffer_gpu)
	, normal_map_gpu(normal_map_gpu)
	, output_resolution( make_int2(std::get<0>(output_resolution), std::get<1>(output_resolution) ) )
	, n_hf_entries(n_hf_entries)
	, buffer_length(buffer_length)
{
}

template<class Primitive>
void Abstract_Intersector<Primitive>::add_primitives(std::vector<Primitive>& primitives)
{
	primitives_cpu = primitives;
	n_primitives = (int)primitives.size();
	presort_primitives();
	primitives_gpu = allocate_primitives_on_gpu(primitives_cpu);
}

template<class Primitive>
void Abstract_Intersector<Primitive>::add_primitives_py(py::array& primitives)
{
	allocate_primitives_cpu(primitives);
	presort_primitives();
	primitives_gpu = allocate_primitives_on_gpu(primitives_cpu);
}

template<class Primitive>
Abstract_Intersector<Primitive>::~Abstract_Intersector<Primitive>()
{
	// todo fix leaks
}

template<class Primitive>
std::pair< py::array_t<float>, py::array_t<float> >  Abstract_Intersector<Primitive>::rasterize_py( float image_plane )
{
	rasterize( image_plane );
	return std::pair<py::array_t<float>, py::array_t<float> >(get_extended_height_field_py(), get_normal_map_py());
}

template<class Primitive>
py::array_t<float> Abstract_Intersector<Primitive>::get_normal_map_py()
{
	auto normal_map_py = create_py_array(output_resolution.x, output_resolution.y, 3);
	cudaMemcpy(normal_map_py.request().ptr, normal_map_gpu, sizeof(float3) * output_resolution.x * output_resolution.y * 1, cudaMemcpyDeviceToHost);
	return normal_map_py;
}

template<class Primitive>
std::vector<float> Abstract_Intersector<Primitive>::get_normal_map()
{
	std::vector<float> normal_map_cpu(3 * output_resolution.x * output_resolution.y );
	cudaMemcpy(&normal_map_cpu[0], normal_map_gpu, sizeof(float3) * output_resolution.x * output_resolution.y * 1, cudaMemcpyDeviceToHost);
	return normal_map_cpu;
}

template<class Primitive>
py::array_t<float> Abstract_Intersector<Primitive>::get_extended_height_field_py()
{
	auto extended_hf_py = create_py_array(output_resolution.x, output_resolution.y, buffer_length * 2);
	cudaMemcpy(extended_hf_py.request().ptr, extended_heightfield_gpu, sizeof(float2) * output_resolution.x * output_resolution.y * buffer_length, cudaMemcpyDeviceToHost);
	return extended_hf_py;
}

template<class Primitive>
void Abstract_Intersector<Primitive>::allocate_primitives_cpu(py::array& primitives)
{
	py::buffer_info info = primitives.request();
	if (info.ndim != 2)
		throw std::invalid_argument("primitives array is expected to be of two dimensions, found "+std::to_string(info.ndim));
	if (info.shape[1] != Primitive::N_FLOAT_PARAMS)
		throw std::invalid_argument("primitives array is expected to be of dimensions nx" + std::to_string(Primitive::N_FLOAT_PARAMS) + ", found " + std::to_string(info.shape[0]) + "x" + std::to_string(info.shape[1]) );
	if (info.format != "f")
		throw std::invalid_argument("spheres array is expected to be of dtype float32, found " + info.format);
	n_primitives = info.shape[0];
	primitives_cpu.resize(n_primitives);
	float* ptr = (float*) info.ptr;
	for (size_t i = 0; i < n_primitives; i++)
	{
		for (size_t j = 0; j < Primitive::N_FLOAT_PARAMS; j++)
		{
			float* primitive_ptr = (float*) &primitives_cpu[i];
			primitive_ptr[j] = *(ptr++);
		}
	}
}

template<class Primitive>
Primitive* Abstract_Intersector<Primitive>::allocate_primitives_on_gpu( const std::vector<Primitive>& primitives_cpu )
{
	Primitive* ptr_gpu;
	cudaMalloc((void**)&ptr_gpu, sizeof(Primitive) * n_primitives);
	cudaMemcpy(ptr_gpu, &primitives_cpu[0], sizeof(Primitive) * n_primitives, cudaMemcpyHostToDevice);
	return ptr_gpu;
}

template<class Primitive>
void Abstract_Intersector<Primitive>::presort_primitives()
{
	assign_aabb();
	if (primitives_cpu.size() == 0)
		throw std::runtime_error("no primitives in call to presort");
	std::sort(primitives_cpu.begin(), primitives_cpu.end(), primitives_cpu[0]);
}

#include "sphere.h"
#include "cylinder.h"
template class Abstract_Intersector<Sphere>;
template class Abstract_Intersector<Cylinder>;