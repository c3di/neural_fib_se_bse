#pragma once

#include "Abstract_Intersector.h"
#include "cuda_utils.h"
#include "cuda_matrix.h"
#include <iostream>


class Volume_Intersector : public Intersector {

public:
	Volume_Intersector(std::tuple<int, int, int> volume_size_,float threshold, float2* extended_heightfield_gpu,float3* normal_map_gpu, int n_hf_entries, int max_buffer_length = 64);
	~Volume_Intersector();

	void allocate_volume_data_cpu(py::array& data);
	float* allocate_volume_data_gpu(const std::vector<float>& volume_data_cpu);
	void allocate_volume_data_gpu_texture(const std::vector<float>& volume_data_cpu);
	void add_volume_py(py::array& data);
	virtual void intersect(float image_plane, GPUMappedFloatBuffer& z_buffer) override;

	virtual std::tuple< py::array_t<float>, py::array_t<float> > intersect_py(float image_plane, GPUMappedFloatBuffer& z_buffer) override;
	virtual py::array_t<float3> get_normal_map_py() override;
	virtual float3* get_normal_map() override;
	virtual py::array_t<float> get_extended_height_field_py() override;



private:

	inline int get_volume_index(int x, int y, int z);
	inline cudaMemcpy3DParms create_copy_params_struct(const cudaExtent volume_size);
	inline cudaResourceDesc create_resource_descriptor();
	inline cudaTextureDesc create_texture_descriptor();
	int3 volume_size;
	int size_of_volume;
	float threshold_value = 0.125;
	std::vector<float> volume_data_cpu;
	float* volume_data_gpu;
	cudaArray* volume_array_gpu = nullptr;
	cudaTextureObject_t volume_data_gpu_tex = 0;


	GPUMappedFloat2Buffer* extended_heightfield;
	GPUMappedFloat3Buffer* normal_map;

	int n_hf_entries;
	int buffer_length = 64;

};