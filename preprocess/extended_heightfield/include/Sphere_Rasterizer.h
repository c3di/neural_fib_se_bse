#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "python_utils.h"

#include <vector>
#include <tuple>

#include "sphere.h"
#include "Abstract_Rasterizer.h"

class Sphere_Rasterizer : public Abstract_Rasterizer<Sphere>
{
public:
	Sphere_Rasterizer(std::pair<int, int> output_resolution, int n_hf_entries, int max_buffer_length = 64);
	Sphere_Rasterizer(float2* extended_heightfield_gpu, float* z_buffer_gpu, float3* normal_map_gpu, std::pair<int, int> output_resolution, int n_hf_entries, int max_buffer_length = 64);
	virtual ~Sphere_Rasterizer();

	virtual void rasterize( float image_plane ) override;

protected:
	virtual void assign_aabb() override;
};
