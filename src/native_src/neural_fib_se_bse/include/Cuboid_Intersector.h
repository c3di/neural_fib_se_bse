#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "python_utils.h"

#include <vector>
#include <tuple>

#include "cuboid.h"
#include "Abstract_Intersector.h"

class Cuboid_Intersector : public Abstract_Intersector<Cuboid>
{
public:
	Cuboid_Intersector(std::tuple<int, int> output_resolution, int n_hf_entries, int max_buffer_length = 64);
	Cuboid_Intersector(float2* extended_heightfield_gpu, float* z_buffer_gpu, float3* normal_map_gpu, std::tuple<int, int> output_resolution, int n_hf_entries, int max_buffer_length = 64);
	virtual ~Cuboid_Intersector();

	virtual void intersect( float image_plane ) override;

protected:
	virtual void assign_aabb() override;

};
