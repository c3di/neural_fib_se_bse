#pragma once

#include "Volume_Intersector.h"

//------------ CUDA Code -----------------------//


__device__ inline static float get_tex_pos_value(cudaTextureObject_t volume_texture, float x, float y, float z) {
	return tex3D<float>(volume_texture, z, y, x);
}

__device__ inline static float get_x_differential_texture(cudaTextureObject_t volume_texture, float3 normalized_pos, float eps_x) {
	return (get_tex_pos_value(volume_texture, normalized_pos.x + eps_x, normalized_pos.y, normalized_pos.z) 
		- get_tex_pos_value(volume_texture, normalized_pos.x - eps_x, normalized_pos.y, normalized_pos.z)) 
		/ (2.0f * eps_x);
}

__device__ inline static float get_y_differential_texture(cudaTextureObject_t volume_texture, float3 normalized_pos, float eps_y) {
	return (get_tex_pos_value(volume_texture, normalized_pos.x, normalized_pos.y + eps_y, normalized_pos.z) 
		- get_tex_pos_value(volume_texture, normalized_pos.x, normalized_pos.y - eps_y, normalized_pos.z)) 
		/ (2.0f * eps_y);
}

__device__ inline static float get_z_differential_texture(cudaTextureObject_t volume_texture, float3 normalized_pos, float eps_z) {
	return (get_tex_pos_value(volume_texture, normalized_pos.x, normalized_pos.y, normalized_pos.z + eps_z) 
		- get_tex_pos_value(volume_texture, normalized_pos.x, normalized_pos.y, normalized_pos.z - eps_z)) 
		/ (2.0f * eps_z);
}



__global__ void get_height_field_marching_volume_kernel(cudaTextureObject_t volume_texture, float density_threshold, float2* extended_heightfield, int3 volume_size, int buffer_length, int n_hf_entries,
	float image_plane_z) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx >= volume_size.x)
		return;
	if (idy >= volume_size.y)
		return;

	int pixel_index = idy * volume_size.x + idx;

	int hit_index = 0;
	int buffer_offset = pixel_index * buffer_length;


	while (hit_index < buffer_length && extended_heightfield[buffer_offset + hit_index] != empty_interval)
	{
		hit_index++;
	}
	if (hit_index >= buffer_length) {
		return;
	}

	int num_hf_entries = 0;

	float norm_x = ((float)idx + 0.5f) / volume_size.x;
	float norm_y = ((float)idy + 0.5f) / volume_size.y;
	float norm_z = image_plane_z / volume_size.z;

	float step = 0.5f / volume_size.z;

	bool found_entry = false;
	float entry = -1.0f;
	float exit = -1.0f;

	for (; norm_z <= 1.0f && (num_hf_entries < n_hf_entries && hit_index < buffer_length); norm_z += step) {
		float density = get_tex_pos_value(volume_texture, norm_x, norm_y, norm_z);
		
		bool hit_detected = density > density_threshold;

		if (hit_detected && !found_entry) {
				entry = norm_z;
				found_entry = true;
			
		} else if(!hit_detected && found_entry)
		{
				exit = norm_z;
				found_entry = false;
				extended_heightfield[buffer_offset + hit_index] = make_float2(entry, exit);
				num_hf_entries++;
				hit_index++;
			
		}

	}
	if (found_entry && hit_index < buffer_length) {
		exit = norm_z - step;
		extended_heightfield[buffer_offset + hit_index] = make_float2(entry, exit);
	}


}


__global__ void get_normal_map_single_kernel_marching_volume_kernel(cudaTextureObject_t volume_texture, float density_threshold, float3* normal_map, float* z_buffer, int3 volume_size, float image_plane_z) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx >= volume_size.x)
		return;
	if (idy >= volume_size.y)
		return;

	int pixel_index = idy * volume_size.x + idx;

	float norm_x = ((float)idx + 0.5f) / volume_size.x;
	float norm_y = ((float)idy + 0.5f) / volume_size.y;
	float norm_z = image_plane_z / volume_size.z;
	float step = 0.5f / volume_size.z;

	float previous_z = norm_z;
	float previous_density = get_tex_pos_value(volume_texture, norm_x, norm_y, norm_z);
	norm_z += step;


	for (; norm_z <= 1.0f; norm_z += step) {
		float density = get_tex_pos_value(volume_texture, norm_x, norm_y, norm_z);

		if ((previous_density - density_threshold) * (density - density_threshold) < 0.0f) {

			float alpha = (density_threshold - previous_density) / (density - previous_density);
			float surface = previous_z + alpha * step;
			z_buffer[pixel_index] = surface;

			float3 normalized_pos = make_float3(norm_x, norm_y, surface);

			float dx = get_x_differential_texture(volume_texture, normalized_pos, 1.0f / volume_size.x);

			float dy = get_y_differential_texture(volume_texture, normalized_pos, 1.0f / volume_size.y);

			float dz = get_z_differential_texture(volume_texture, normalized_pos, 1.0f / volume_size.z);
			float3 normal = getNormalizedVec(make_float3(dx, dy, dz));
			normal.z = normal.z < 0.0f ? (normal.z * -1.0f) : normal.z;
				
			normal_map[pixel_index] = normal;

			return;
		}
		previous_density = density;
		previous_z = norm_z;
	}
};
//------------ CPP Code -----------------------//


Volume_Intersector::Volume_Intersector(std::tuple<int, int, int> volume_size_,
										float threshold,
										float2* extended_heightfield_gpu,
										float3* normal_map_gpu, 
										int n_hf_entries, 
										int max_buffer_length) 
	: volume_size(as_int3(volume_size_)),threshold_value(threshold), n_hf_entries(n_hf_entries), buffer_length(max_buffer_length){
	this->size_of_volume = this->volume_size.x * this->volume_size.y * this->volume_size.z;
	extended_heightfield = new GPUMappedFloat2Buffer(make_int3(this->volume_size.x, this->volume_size.y, buffer_length), extended_heightfield_gpu);
	normal_map = new GPUMappedFloat3Buffer(make_int3(this->volume_size.y, this->volume_size.x, 1), normal_map_gpu);
};
	

Volume_Intersector::~Volume_Intersector() {
	if (this->volume_data_gpu_tex) {
		cudaDestroyTextureObject(this->volume_data_gpu_tex);
		this->volume_data_gpu_tex = 0;
	}
	if (this->volume_array_gpu) {
		cudaFreeArray(this->volume_array_gpu);
		this->volume_array_gpu = nullptr;
	}
	if (this->extended_heightfield) {
		delete(this->extended_heightfield);
	}
	if (this->normal_map) {
		delete(this->normal_map);
	}
}


void Volume_Intersector::allocate_volume_data_gpu_texture(py::array& volume_data) {
	py::buffer_info info = volume_data.request();
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

	float* ptr = (float*)info.ptr;
	//std::memcpy(volume_data_cpu.data(), info.ptr, this->size_of_volume * sizeof(float));

	cudaExtent volume_size = make_cudaExtent(this->volume_size.z, this->volume_size.y, this->volume_size.x);

	cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();

	cudaMalloc3DArray(&this->volume_array_gpu, &channel_desc, volume_size);

	cudaMemcpy3DParms copy_params = create_copy_params_struct(ptr, volume_size);
	cudaMemcpy3D(&copy_params);

	cudaResourceDesc res_desc = create_resource_descriptor();

	cudaTextureDesc tex_desc = create_texture_descriptor();

	cudaCreateTextureObject(&this->volume_data_gpu_tex, &res_desc, &tex_desc, nullptr);
}


inline cudaMemcpy3DParms Volume_Intersector::create_copy_params_struct(float* volume_data, const cudaExtent volume_size) {
	cudaMemcpy3DParms copy_params = { 0 };
	copy_params.srcPtr = make_cudaPitchedPtr((void*)volume_data, this->volume_size.z * sizeof(float), this->volume_size.z, this->volume_size.y);
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
	this->allocate_volume_data_gpu_texture(data);
}


void Volume_Intersector::intersect(float image_plane, GPUMappedFloatBuffer& z_buffer) {
		
	int2 grid_size = make_int2(this->volume_size.x, this->volume_size.y);
	dim3 block_size(32, 8);
	dim3 num_blocks((grid_size.x + block_size.x - 1) / block_size.x, (grid_size.y + block_size.y - 1) / block_size.y);

	get_height_field_marching_volume_kernel << <num_blocks, block_size >> > (this->volume_data_gpu_tex, this->threshold_value, this->extended_heightfield->gpu_ptr(), this->volume_size, this->buffer_length, this->n_hf_entries, image_plane);

	get_normal_map_single_kernel_marching_volume_kernel << <num_blocks, block_size >> > (this->volume_data_gpu_tex, this->threshold_value, this->normal_map->gpu_ptr(), z_buffer.gpu_ptr(), this->volume_size, image_plane);

	throw_on_cuda_error();
}


 std::tuple< py::array_t<float>, py::array_t<float> > Volume_Intersector::intersect_py(float image_plane, GPUMappedFloatBuffer& z_buffer) {
	 intersect(image_plane, z_buffer);
	 return std::tuple<py::array_t<float>, py::array_t<float>>(get_extended_height_field_py(), get_normal_map_py());
}

 py::array_t<float3> Volume_Intersector::get_normal_map_py() {
	normal_map->pull_from_gpu();
	return normal_map->as_py();
 }

 float3* Volume_Intersector::get_normal_map() {
	normal_map->pull_from_gpu();
	return normal_map->cpu_ptr();
 }

 py::array_t<float> Volume_Intersector::get_extended_height_field_py() {
	extended_heightfield->pull_from_gpu();
	return extended_heightfield->as_py();
 }


