#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "python_utils.h"

#include <vector>
#include <tuple>

#include "cylinder.h"
#include "Abstract_Rasterizer.h"

class Cylinder_Rasterizer : public Abstract_Rasterizer<Cylinder>
{
public:
	Cylinder_Rasterizer(std::pair<int, int> output_resolution, int n_hf_entries, int max_buffer_length = 64);
	Cylinder_Rasterizer(float2* extended_heightfield_gpu, float* z_buffer_gpu, float3* normal_map_gpu, std::pair<int, int> output_resolution, int n_hf_entries, int max_buffer_length = 64);
	virtual ~Cylinder_Rasterizer();

	virtual void rasterize( float image_plane ) override;

protected:
	virtual void assign_aabb() override;

};
