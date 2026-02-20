#pragma once

#include "Volume_Intersector.h"

//------------ CUDA Code -----------------------//
__global__ void intersect_volume_kernel(cudaTextureObject_t volume_texture, float density_threshold, float2* extended_heightfield, float3* normal_map, float* z_buffer, int3 volume_size, int buffer_length, int n_hf_entries, 
							float image_plane_z, bool debug, int2 debug_position ) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx >= volume_size.x)
		return;
	if (idy >= volume_size.y)
		return;

	int pixel_index = idy * volume_size.x + idx;

	int hit_index = 0;

	while (extended_heightfield[pixel_index * buffer_length + hit_index] != empty_interval)
	{
		if (debug && idx == debug_position.x && idy == debug_position.y)
		{
			float2 value = extended_heightfield[pixel_index * buffer_length + hit_index];
			printf("  hit index %i %.2f %.2f\n", hit_index, value.x, value.y);
		}
		hit_index++;
		if (hit_index >= buffer_length)
			return;
	}


	const float pixel_x = (float) idx;
	const float pixel_y = (float) idy;

	// ray for intersection
	float3 ray_origin = make_float3(0.0f, 0.0f, 0.0f);
	float3 ray_direction = make_float3(0.0f, 0.0f, 1.0f); // moving only in z direction
	float voxel_z = 0;
	float3 normalized_pos = make_float3(pixel_x / volume_size.x, pixel_y / volume_size.y, 0.0f); // use order z,y,x to access texture data

	float t = 0.0f;
	float t_max = 1.0f;
	float step = 1.0f / volume_size.z;

	bool found_entry = false;
	float entry;
	float exit;

	while (t < t_max) {
		normalized_pos.z = ray_origin.z + t * ray_direction.z;


		float density = tex3D<float>(volume_texture, normalized_pos.z, normalized_pos.y, normalized_pos.x);

		if (density > density_threshold) {

			if (!found_entry) {
				entry = normalized_pos.z;
				found_entry = true;
			}
		}
		else {
			if (found_entry) {
				exit = normalized_pos.z;
				found_entry = false;
				extended_heightfield[pixel_index * buffer_length + hit_index] = make_float2(entry, exit);
				hit_index++;
			}
		}
		if (hit_index >= buffer_length) {
			return;
		}
		t += step;
	}
	if (found_entry && hit_index < buffer_length) {
		exit = normalized_pos.z;
		extended_heightfield[pixel_index * buffer_length + hit_index] = make_float2(entry, exit);
	}


};


//------------ CPP Code -----------------------//


Volume_Intersector::Volume_Intersector(std::tuple<int, int, int> volume_size_,
										float2* extended_heightfield_gpu,
										float3* normal_map_gpu, 
										int n_hf_entries, 
										int max_buffer_length) 
	: volume_size(as_int3(volume_size_)), n_hf_entries(n_hf_entries), buffer_length(max_buffer_length) {
	this->size_of_volume = this->volume_size.x * this->volume_size.y * this->volume_size.z;
	this->volume_data_cpu.resize(this->size_of_volume);
	this->volume_data_gpu = nullptr;
	extended_heightfield = new GPUMappedFloat2Buffer(make_int3(this->volume_size.x, this->volume_size.y, buffer_length), extended_heightfield_gpu);
	normal_map = new GPUMappedFloat3Buffer(make_int3(this->volume_size.x, this->volume_size.y, 1), normal_map_gpu);
};
	

Volume_Intersector::~Volume_Intersector() {
	if (this->volume_data_gpu) {
		cudaFree(this->volume_data_gpu);
	}
	if (this->volume_data_gpu_tex) {
		cudaDestroyTextureObject(this->volume_data_gpu_tex);
		this->volume_data_gpu_tex = 0;
	}
	if (this->volume_array_gpu) {
		cudaFreeArray(this->volume_array_gpu);
		this->volume_array_gpu = nullptr;
	}
}



void Volume_Intersector::allocate_volume_data_cpu(py::array& data)
{
	py::buffer_info info = data.request();
	if (info.ndim != 3) {
		throw std::invalid_argument("data array is expected to be of three dimensions, found " + std::to_string(info.ndim));
	}
	if (info.shape[0] != this->volume_size.x) {
		throw std::invalid_argument("data array is expected to be of dimensions nx: " + std::to_string(this->volume_size.x) + ", found x: " + std::to_string(info.shape[0]));
	}
	if (info.shape[1] != this->volume_size.y) {
		throw std::invalid_argument("data array is expected to be of dimensions ny: " + std::to_string(this->volume_size.y) + ", found y: " + std::to_string(info.shape[1]));
	}
	if (info.shape[2] != this->volume_size.z) {
		throw std::invalid_argument("data array is expected to be of dimensions nz: " + std::to_string(this->volume_size.z) + ", found z: " + std::to_string(info.shape[2]));
	}

	if (info.format != "f") {
		throw std::invalid_argument("data array is expected to be of dtype float32, found " + info.format);
	}
	float* ptr = (float*) info.ptr;
	std::memcpy(volume_data_cpu.data(), info.ptr, this->size_of_volume * sizeof(float));

}


inline int Volume_Intersector::get_volume_index(int x, int y, int z) {
	return x * (volume_size.y * volume_size.z)
		+ y * volume_size.z
		+ z;
};


float* Volume_Intersector::allocate_volume_data_gpu(const std::vector<float>& volume_data_cpu)
{
	float* ptr_gpu;
	cudaMalloc((void**)&ptr_gpu, sizeof(float) * this->size_of_volume);
	cudaMemcpy(ptr_gpu, &this->volume_data_cpu[0], sizeof(float) * this->size_of_volume, cudaMemcpyHostToDevice);
	return ptr_gpu;
}

void Volume_Intersector::allocate_volume_data_gpu_texture(const std::vector<float>& volume_data_cpu) {
	cudaExtent volume_size = make_cudaExtent(this->volume_size.z, this->volume_size.y, this->volume_size.x);

	cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();

	cudaMalloc3DArray(&this->volume_array_gpu, &channel_desc, volume_size);

	cudaMemcpy3DParms copy_params = create_copy_params_struct(volume_size);
	cudaMemcpy3D(&copy_params);

	cudaResourceDesc res_desc = create_resource_descriptor();

	cudaTextureDesc tex_desc = create_texture_descriptor();

	cudaCreateTextureObject(&this->volume_data_gpu_tex, &res_desc, &tex_desc, nullptr);
}


inline cudaMemcpy3DParms Volume_Intersector::create_copy_params_struct(const cudaExtent volume_size) {
	cudaMemcpy3DParms copy_params = { 0 };
	copy_params.srcPtr = make_cudaPitchedPtr((void*)volume_data_cpu.data(), this->volume_size.z * sizeof(float), this->volume_size.z, this->volume_size.y);
	copy_params.dstArray = this->volume_array_gpu;
	copy_params.extent = make_cudaExtent(
		this->volume_size.z,
		this->volume_size.y,
		this->volume_size.x
	);
	copy_params.kind = cudaMemcpyHostToDevice;

	return copy_params;
}


inline cudaResourceDesc Volume_Intersector::create_resource_descriptor() {
	cudaResourceDesc res_desc= {};
	res_desc.resType = cudaResourceTypeArray;
	res_desc.res.array.array = this->volume_array_gpu;

	return res_desc;
};


inline cudaTextureDesc Volume_Intersector::create_texture_descriptor() {
	cudaTextureDesc tex_desc = {};
	tex_desc.addressMode[0] = cudaAddressModeClamp;
	tex_desc.addressMode[1] = cudaAddressModeClamp;
	tex_desc.addressMode[2] = cudaAddressModeClamp;
	tex_desc.filterMode = cudaFilterModeLinear;
	tex_desc.readMode = cudaReadModeElementType;
	tex_desc.normalizedCoords = 1;

	return tex_desc;
};

void Volume_Intersector::add_volume_py(py::array& data) {
	this->allocate_volume_data_cpu(data);
	this->allocate_volume_data_gpu_texture(this->volume_data_cpu);
	//this->volume_data_gpu = this->allocate_volume_data_gpu(this->volume_data_cpu);
}


void Volume_Intersector::intersect(float image_plane, GPUMappedFloatBuffer& z_buffer) {
	int2 grid_size = make_int2(this->volume_size.x, this->volume_size.y);
	dim3 block_size(16, 16);
	dim3 num_blocks((grid_size.x + block_size.x - 1) / block_size.x, (grid_size.y + block_size.y - 1) / block_size.y);
	intersect_volume_kernel << <num_blocks, block_size >> > (this->volume_data_gpu_tex, this->threshold_value, this->extended_heightfield->gpu_ptr(), this->normal_map->gpu_ptr(), z_buffer.gpu_ptr(), this->volume_size, this->buffer_length, this->n_hf_entries, image_plane, false, make_int2(425, 425) );
	throw_on_cuda_error();
}


 std::tuple< py::array_t<float>, py::array_t<float> > Volume_Intersector::intersect_py(float image_plane, GPUMappedFloatBuffer& z_buffer) {
	 intersect(image_plane, z_buffer);
	 return std::tuple<py::array_t<float>, py::array_t<float>>(get_extended_height_field_py(), get_normal_map_py());
}

 py::array_t<float3> Volume_Intersector::get_normal_map_py() {
	 std::cout << "Not implemented" << std::endl;
	 auto ret_val = create_py_array(1, 1, 1);
	 return ret_val;
 }

 float3* Volume_Intersector::get_normal_map() {
	std::cout << "Not implemented" << std::endl;
	return &make_float3(0, 0, 0);
 }

 py::array_t<float> Volume_Intersector::get_extended_height_field_py() {
	std::cout << "Not implemented" << std::endl;
	auto ret_val = create_py_array(1, 1, 1);
	return ret_val;
 }


