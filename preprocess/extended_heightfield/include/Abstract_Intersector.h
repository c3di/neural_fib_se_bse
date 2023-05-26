#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "python_utils.h"
#include "gpu_mapped_object.h"

#include <vector>
#include <tuple>

class Intersector
{
public:
	virtual std::tuple< py::array_t<float>, py::array_t<float> > intersect_py(float image_plane) = 0;
	virtual void intersect(float image_plane) = 0;
	virtual py::array_t<float3> get_normal_map_py() = 0;
	virtual float3* get_normal_map() = 0;
	virtual py::array_t<float> get_extended_height_field_py() = 0;
};

template<class Primitive>
class Abstract_Intersector : public Intersector
{
public:
	Abstract_Intersector(std::tuple<int, int> output_resolution, int n_hf_entries, int buffer_length = 64);
	Abstract_Intersector(float2* extended_heightfield_gpu, float* z_buffer_gpu, float3* normal_map_gpu, std::tuple<int, int> output_resolution, int n_hf_entries, int buffer_length = 64);
	virtual ~Abstract_Intersector();

	virtual void add_primitives(std::vector<Primitive>& primitives);
	virtual void add_primitives_py(py::array& primitives);

	virtual std::tuple< py::array_t<float>, py::array_t<float> > intersect_py( float image_plane );
	virtual void intersect( float image_plane ) = 0;

	virtual py::array_t<float3> get_normal_map_py();
	virtual float3* get_normal_map();
	virtual py::array_t<float> get_extended_height_field_py();

protected:
	virtual void allocate_primitives_cpu(py::array& spheres);

	virtual Primitive* allocate_primitives_on_gpu(const std::vector<Primitive>& primitives_cpu);

	virtual void assign_aabb() = 0;
	virtual void presort_primitives();

protected:
	std::vector<Primitive> primitives_cpu;
	Primitive* primitives_gpu;
	int n_primitives;

	GPUMappedFloat2Buffer* extended_heightfield;
	GPUMappedFloat3Buffer* normal_map;
	GPUMappedFloatBuffer* z_buffer;

	int2 output_resolution;
	int n_hf_entries;
	int buffer_length;
	float image_plane;
};
