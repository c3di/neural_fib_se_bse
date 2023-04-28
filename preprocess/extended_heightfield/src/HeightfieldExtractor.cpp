#pragma once

#include "HeightFieldExtractor.h"
#include "Sphere_Rasterizer.h"
#include "CSG_Resolver.h"

#include "cuda_utils.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>

HeightFieldExtractor::HeightFieldExtractor(py::array& spheres, std::pair<int, int> output_resolution, int n_hf_entries, int max_buffer_length )
	: output_resolution( make_int2(std::get<0>(output_resolution), std::get<1>(output_resolution)) )
	, n_hf_entries(n_hf_entries)
	, buffer_length(max_buffer_length)
{
	allocate_extended_heightfield_on_gpu();
	sphere_rasterizer = new Sphere_Rasterizer(extended_heightfield_gpu, spheres, output_resolution, n_hf_entries, max_buffer_length);
	csg_resolver = new CSG_Resolver(extended_heightfield_gpu, make_int3(std::get<0>(output_resolution), std::get<1>(output_resolution), buffer_length), n_hf_entries );
}

HeightFieldExtractor::~HeightFieldExtractor()
{
	delete csg_resolver;
	delete sphere_rasterizer;
}

py::array_t<float> HeightFieldExtractor::rasterize_py(float image_plane)
{
	rasterize( image_plane );
	auto pyarray = create_py_array(output_resolution.x, output_resolution.y, buffer_length * 2);
	cudaMemcpy(pyarray.request().ptr, extended_heightfield_gpu, sizeof(float2) * output_resolution.x * output_resolution.y * buffer_length, cudaMemcpyDeviceToHost);
	return pyarray;
}

void HeightFieldExtractor::rasterize(float image_plane)
{
	sphere_rasterizer->rasterize_spheres(image_plane);
	csg_resolver->resolve_csg(image_plane);
}

float2* HeightFieldExtractor::allocate_extended_heightfield_on_gpu()
{
	float2* ptr_gpu;
	cudaMalloc((void**)&ptr_gpu, sizeof(float2) * output_resolution.x * output_resolution.y * buffer_length);
	return ptr_gpu;
}