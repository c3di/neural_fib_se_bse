#pragma once

#include "Abstract_Rasterizer.h"

#include "cuda_utils.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>

template<class Primitive>
Abstract_Rasterizer<Primitive>::Abstract_Rasterizer<Primitive>(py::array& primitives, std::pair<int, int> output_resolution, int n_hf_entries, int max_buffer_length)
	: output_resolution(make_int2(std::get<0>(output_resolution), std::get<1>(output_resolution)))
	, n_hf_entries(n_hf_entries)
{
	allocate_primitives_cpu(primitives);
	if (n_primitives < max_buffer_length)
		buffer_length = n_primitives;
	else
		buffer_length = max_buffer_length;
	presort_primitives();
	primitives_gpu = allocate_primitives_on_gpu(primitives_cpu);

	extended_heightfield_gpu = allocate_float2_buffer_on_gpu(make_int3(std::get<0>(output_resolution), std::get<1>(output_resolution), buffer_length));
	normal_map_gpu = allocate_float3_buffer_on_gpu(make_int3(std::get<0>(output_resolution), std::get<1>(output_resolution), 1));
	z_buffer_gpu = allocate_float_buffer_on_gpu(make_int3(std::get<0>(output_resolution), std::get<1>(output_resolution), 1));
}

template<class Primitive>
Abstract_Rasterizer<Primitive>::Abstract_Rasterizer<Primitive>(float2* extended_heightfield_gpu, py::array& primitives, std::pair<int, int> output_resolution, int n_hf_entries, int max_buffer_length)
	: extended_heightfield_gpu(extended_heightfield_gpu)
	, output_resolution( make_int2(std::get<0>(output_resolution), std::get<1>(output_resolution) ) )
	, n_hf_entries(n_hf_entries)
{
	allocate_primitives_cpu(primitives);
	if (n_primitives < max_buffer_length)
		buffer_length = n_primitives;
	else
		buffer_length = max_buffer_length;
	presort_primitives();
	primitives_gpu = allocate_primitives_on_gpu(primitives_cpu);
	normal_map_gpu = allocate_float3_buffer_on_gpu(make_int3(std::get<0>(output_resolution), std::get<1>(output_resolution), 1));
	z_buffer_gpu = allocate_float_buffer_on_gpu(make_int3(std::get<0>(output_resolution), std::get<1>(output_resolution), 1));
}

template<class Primitive>
Abstract_Rasterizer<Primitive>::Abstract_Rasterizer<Primitive>(float2* extended_heightfield_gpu, std::vector<Primitive>& primitives, std::pair<int, int> output_resolution, int n_hf_entries, int max_buffer_length)
	: extended_heightfield_gpu(extended_heightfield_gpu)
	, output_resolution(make_int2(std::get<0>(output_resolution), std::get<1>(output_resolution)))
	, n_hf_entries(n_hf_entries)
	, primitives_cpu(primitives)
{
	n_primitives = (int)primitives.size();
	if (n_primitives < max_buffer_length)
		buffer_length = n_primitives;
	else
		buffer_length = max_buffer_length;
	presort_primitives();
	primitives_gpu = allocate_primitives_on_gpu(primitives_cpu);
	normal_map_gpu = allocate_float3_buffer_on_gpu(make_int3(std::get<0>(output_resolution), std::get<1>(output_resolution), 1));
	z_buffer_gpu = allocate_float_buffer_on_gpu(make_int3(std::get<0>(output_resolution), std::get<1>(output_resolution), 1));
}

template<class Primitive>
Abstract_Rasterizer<Primitive>::~Abstract_Rasterizer<Primitive>()
{
	// todo fix leaks
}

template<class Primitive>
std::pair< py::array_t<float>, py::array_t<float> >  Abstract_Rasterizer<Primitive>::rasterize_py( float image_plane )
{
	rasterize( image_plane );
	return std::pair<py::array_t<float>, py::array_t<float> >(get_extended_height_field_py(), get_normal_map_py());
}

template<class Primitive>
py::array_t<float> Abstract_Rasterizer<Primitive>::get_normal_map_py()
{
	auto normal_map_py = create_py_array(output_resolution.x, output_resolution.y, 3);
	cudaMemcpy(normal_map_py.request().ptr, normal_map_gpu, sizeof(float3) * output_resolution.x * output_resolution.y * 1, cudaMemcpyDeviceToHost);
	return normal_map_py;
}

template<class Primitive>
std::vector<float> Abstract_Rasterizer<Primitive>::get_normal_map()
{
	std::vector<float> normal_map_cpu(3 * output_resolution.x * output_resolution.y );
	cudaMemcpy(&normal_map_cpu[0], normal_map_gpu, sizeof(float3) * output_resolution.x * output_resolution.y * 1, cudaMemcpyDeviceToHost);
	return normal_map_cpu;
}

template<class Primitive>
py::array_t<float> Abstract_Rasterizer<Primitive>::get_extended_height_field_py()
{
	auto extended_hf_py = create_py_array(output_resolution.x, output_resolution.y, buffer_length * 2);
	cudaMemcpy(extended_hf_py.request().ptr, extended_heightfield_gpu, sizeof(float2) * output_resolution.x * output_resolution.y * buffer_length, cudaMemcpyDeviceToHost);
	return extended_hf_py;
}

template<class Primitive>
void Abstract_Rasterizer<Primitive>::allocate_primitives_cpu(py::array& primitives)
{
	py::buffer_info info = primitives.request();
	if (info.ndim != 2)
		throw std::invalid_argument("spheres array is expected to be of dimensions nx4");
	if (info.shape[1] != 4)
		throw std::invalid_argument("spheres array is expected to be of dimensions nx4");
	if (info.format != "f")
		throw std::invalid_argument("spheres array is expected to be of dtype float32, found " + info.format);
	n_primitives = info.shape[0];
	primitives_cpu.resize(n_primitives);
	float* ptr = (float*) info.ptr;
	const size_t size_in_float = sizeof(Primitive) / sizeof(float);
	for (size_t i = 0; i < info.shape[0]; i++)
	{
		for (size_t j = 0; j < size_in_float; j++)
		{
			float* primitive_ptr = (float*) &primitives_cpu[i];
			primitive_ptr[j] = *(ptr++);
		}
	}
}

template<class Primitive>
Primitive* Abstract_Rasterizer<Primitive>::allocate_primitives_on_gpu( const std::vector<Primitive>& primitives_cpu )
{
	Primitive* ptr_gpu;
	cudaMalloc((void**)&ptr_gpu, sizeof(Primitive) * n_primitives);
	cudaMemcpy(ptr_gpu, &primitives_cpu[0], sizeof(Primitive) * n_primitives, cudaMemcpyHostToDevice);
	return ptr_gpu;
}

template<class Primitive>
void Abstract_Rasterizer<Primitive>::presort_primitives()
{
	if (primitives_cpu.size() == 0)
		throw std::runtime_error("no primitives in call to presort");
	std::sort(primitives_cpu.begin(), primitives_cpu.end(), primitives_cpu[0]);
}

#include "sphere.h"

template class Abstract_Rasterizer<Sphere>;