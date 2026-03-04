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
	float3 normalized_pos = make_float3((pixel_x + 0.5f) / volume_size.x, (pixel_y + 0.5f) / volume_size.y, 0.0f); // use order z,y,x to access texture data

	float t = 0.0f;
	float t_max = 1.0f;
	float step = 0.1f / volume_size.z;

	bool found_entry = false;
	float entry;
	float exit;

	//Heightfield
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


	//Normal map

	normalized_pos.z = 0.0f;

	float previous_z = 0.0f;
	float previous_density = tex3D<float>(volume_texture, 0.0f, normalized_pos.y, normalized_pos.x);

	for (float z = step; z <= 1.0f; z += step) {
		float density = tex3D<float>(volume_texture, z, normalized_pos.y, normalized_pos.x);

		//if (previous_density < density && density > density_threshold) {
		if((previous_density - density_threshold) * (density - density_threshold) < 0.0f){
			float alpha = (density_threshold - previous_density) / (density - previous_density);

			float surface = previous_z + alpha * step;

			float eps_x = 1.0f / volume_size.x;
			float eps_y = 1.0f / volume_size.y;
			float eps_z = 1.0f / volume_size.z;

			float dx = (tex3D<float>(volume_texture, surface, normalized_pos.y, normalized_pos.x + eps_x) -
				tex3D<float>(volume_texture, surface, normalized_pos.y, normalized_pos.x - eps_x))
				/ (2.0f * eps_x);

			float dy = (tex3D<float>(volume_texture, surface, normalized_pos.y + eps_y, normalized_pos.x) -
				tex3D<float>(volume_texture, surface, normalized_pos.y - eps_y, normalized_pos.x))
				/ (2.0f * eps_y);

			float dz = (tex3D<float>(volume_texture, surface + eps_z, normalized_pos.y, normalized_pos.x) -
				tex3D<float>(volume_texture, surface - eps_z, normalized_pos.y, normalized_pos.x))
				/ (2.0f * eps_z);

			float length = dx * dx + dy * dy + dz * dz;
			if (length > 1e-10f) {
				float inv = rsqrtf(length);
				float3 grad = make_float3(-dx, -dy, -dz);
				if (grad.z < 0.0f) {
					grad.z *= -1.0f;
				}
				if (length > 0.0f) {
					grad.x *= inv;
					grad.y *= inv;
					grad.z *= inv;
				}

				if (z_buffer[pixel_index] > surface) {
					z_buffer[pixel_index] = surface;
					
					normal_map[pixel_index] = make_float3(grad.x , grad.y, grad.z );
				}

			}
			else {
				normal_map[pixel_index] = make_float3(0.55f, .55f, .55f);
			}

			return;
		}
		previous_density = density;
		previous_z = z;
	}

};


__device__ inline static float get_tex_pos_value(cudaTextureObject_t volume_texture, float x, float y, float z) {
	return tex3D<float>(volume_texture, z, y, x);
}

__device__ inline static float get_x_differential_texture(cudaTextureObject_t volume_texture, float3 normalized_pos, float surface, float eps_x) {
	return (get_tex_pos_value(volume_texture, normalized_pos.x + eps_x, normalized_pos.y, surface) 
		- get_tex_pos_value(volume_texture, normalized_pos.x - eps_x, normalized_pos.y, surface)) 
		/ (2.0f * eps_x);
}

__device__ inline static float get_y_differential_texture(cudaTextureObject_t volume_texture, float3 normalized_pos, float surface, float eps_y) {
	return (get_tex_pos_value(volume_texture, normalized_pos.x, normalized_pos.y + eps_y, surface) 
		- get_tex_pos_value(volume_texture, normalized_pos.x, normalized_pos.y - eps_y, surface)) 
		/ (2.0f * eps_y);
}

__device__ inline static float get_z_differential_texture(cudaTextureObject_t volume_texture, float3 normalized_pos, float surface, float eps_z) {
	return (get_tex_pos_value(volume_texture, normalized_pos.x, normalized_pos.y, surface + eps_z) 
		- get_tex_pos_value(volume_texture, normalized_pos.x, normalized_pos.y, surface - eps_z)) 
		/ (2.0f * eps_z);
}


__global__ void intersect_volume_kernel_single_loop(cudaTextureObject_t volume_texture, float density_threshold, float2* extended_heightfield, float3* normal_map, float* z_buffer, int3 volume_size, int buffer_length, int n_hf_entries, 
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

	int num_hf_entries = 0;

	const float pixel_x = (float) idx;
	const float pixel_y = (float) idy;

	float normalized_image_plane_z = image_plane_z / volume_size.z;
	float3 normalized_pos = make_float3((pixel_x + 0.5f) / volume_size.x, (pixel_y + 0.5f) / volume_size.y, normalized_image_plane_z); // use order z,y,x to access texture data

	float step = 0.1f / volume_size.z;

	bool found_entry = false;
	float entry = -1.0f;
	float exit = -1.0f;


	bool n_map_written = false;
	
	float previous_z = normalized_pos.z;
	float previous_density = get_tex_pos_value(volume_texture, normalized_pos.x, normalized_pos.y, normalized_pos.z);

	float3 eps = make_float3(1.0f / volume_size.x, 1.0f / volume_size.y, 1.0f / volume_size.z);

	for (float m_z = normalized_pos.z + step; m_z <= 1.0f; m_z += step) {
		float density = get_tex_pos_value(volume_texture, normalized_pos.x, normalized_pos.y, m_z);


		if (density > density_threshold) {
			// Heightfield
			if (!found_entry) {
				entry = m_z;
				found_entry = true;
			}

		} else 
		{
			//Heightfield
			if (found_entry) {
				if (entry < 0.0f) {
					previous_density = density;
					previous_z = m_z;
					found_entry = false;
					continue;
				}
				exit = m_z;
				found_entry = false;
				extended_heightfield[pixel_index * buffer_length + hit_index] = make_float2(entry, exit);
				num_hf_entries++;
				hit_index++;
			}
		}
		if (num_hf_entries >= n_hf_entries	|| hit_index >= buffer_length) {
			return;
		}


		if ((previous_density - density_threshold) * (density - density_threshold) < 0.0f) {

						
			// Normal map
			if (!n_map_written) {
				float alpha = (density_threshold - previous_density) / (density - previous_density);

				float surface = previous_z + alpha * step;

				float dx = get_x_differential_texture(volume_texture, normalized_pos, surface, eps.x);

				float dy = get_y_differential_texture(volume_texture, normalized_pos, surface, eps.y);

				float dz = get_z_differential_texture(volume_texture, normalized_pos, surface, eps.z);
				float3 normal = getNormalizedVec(make_float3(dx, dy, dz));
				if (normal.z < 0.0f) {
						normal.z *= -1.0f;
				}
				
				if (z_buffer[pixel_index] > surface) {
					z_buffer[pixel_index] = surface;

					normal_map[pixel_index] = normal;
				}


				n_map_written = true;
			}
		}

		previous_density = density;
		previous_z = m_z;
	}
	if (found_entry && hit_index < buffer_length) {
		exit = previous_z;
		extended_heightfield[pixel_index * buffer_length + hit_index] = make_float2(entry, exit);
	}
};




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

	for (norm_z; norm_z <= 1.0f && (num_hf_entries < n_hf_entries && hit_index < buffer_length); norm_z += step) {
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




__global__ void get_first_z_hit_marching_volume_kernel(cudaTextureObject_t volume_texture, float density_threshold, float* z_buffer, int3 volume_size, float image_plane_z) {
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



	bool searching = true;


	for (norm_z; norm_z <= 1.0f && searching; norm_z += step) {
		float density = get_tex_pos_value(volume_texture, norm_x, norm_y, norm_z);

		if ((previous_density - density_threshold) * (density - density_threshold) < 0.0f) {

			float alpha = (density_threshold - previous_density) / (density - previous_density);
			float surface = previous_z + alpha * step;
			z_buffer[pixel_index] = z_buffer[pixel_index] > surface ? surface : z_buffer[pixel_index];

			searching = false;
		}
		previous_density = density;
		previous_z = norm_z;
	}
};


__global__ void get_normal_map_marching_volume_kernel(cudaTextureObject_t volume_texture, float density_threshold, float3* normal_map, float* z_buffer, int3 volume_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx >= volume_size.x)
		return;
	if (idy >= volume_size.y)
		return;

	int pixel_index = idy * volume_size.x + idx;
	float surface = z_buffer[pixel_index];
	if (surface == empty) {
		return;
	}
	float3 normalized_pos = make_float3(((float)idx + 0.5f) / volume_size.x, ((float)idy + 0.5f) / volume_size.y, z_buffer[pixel_index]);
	float step = 0.5f / volume_size.z;

	float dx = get_x_differential_texture(volume_texture, normalized_pos, surface, 1.0f / volume_size.x);

	float dy = get_y_differential_texture(volume_texture, normalized_pos, surface, 1.0f / volume_size.y);

	float dz = get_z_differential_texture(volume_texture, normalized_pos, surface, 1.0f / volume_size.z);
	float3 normal = getNormalizedVec(make_float3(dx, dy, dz));
	normal.z = normal.z < 0.0f ? (normal.z * -1.0f) : normal.z;
				
	normal_map[pixel_index] = normal;

};




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

	bool searching = true;

	for (norm_z; norm_z <= 1.0f && searching; norm_z += step) {
		float density = get_tex_pos_value(volume_texture, norm_x, norm_y, norm_z);

		if ((previous_density - density_threshold) * (density - density_threshold) < 0.0f) {

			float alpha = (density_threshold - previous_density) / (density - previous_density);
			float surface = previous_z + alpha * step;
			z_buffer[pixel_index] = surface;

			float3 normalized_pos = make_float3(norm_x, norm_y, norm_z);

			float dx = get_x_differential_texture(volume_texture, normalized_pos, surface, 1.0f / volume_size.x);

			float dy = get_y_differential_texture(volume_texture, normalized_pos, surface, 1.0f / volume_size.y);

			float dz = get_z_differential_texture(volume_texture, normalized_pos, surface, 1.0f / volume_size.z);
			float3 normal = getNormalizedVec(make_float3(dx, dy, dz));
			normal.z = normal.z < 0.0f ? (normal.z * -1.0f) : normal.z;
				
			normal_map[pixel_index] = normal;

			searching = false;
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
										int max_buffer_length,
										Implementation impl) 
	: volume_size(as_int3(volume_size_)),threshold_value(threshold), n_hf_entries(n_hf_entries), buffer_length(max_buffer_length), impl(impl){
	this->size_of_volume = this->volume_size.x * this->volume_size.y * this->volume_size.z;
	this->volume_data_gpu = nullptr;
	extended_heightfield = new GPUMappedFloat2Buffer(make_int3(this->volume_size.x, this->volume_size.y, buffer_length), extended_heightfield_gpu);
	normal_map = new GPUMappedFloat3Buffer(make_int3(this->volume_size.y, this->volume_size.x, 1), normal_map_gpu);
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
	if (this->extended_heightfield) {
		delete(this->extended_heightfield);
	}
	if (this->normal_map) {
		delete(this->normal_map);
	}
}

inline int Volume_Intersector::get_volume_index(int x, int y, int z) {
	return x * (volume_size.y * volume_size.z)
		+ y * volume_size.z
		+ z;
};

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
	//this->allocate_volume_data_cpu(data);
	this->allocate_volume_data_gpu_texture(data);
	//this->volume_data_gpu = this->allocate_volume_data_gpu(this->volume_data_cpu);
}


void Volume_Intersector::intersect(float image_plane, GPUMappedFloatBuffer& z_buffer) {
	int minGridSize;
	int blockSize;

	int blockX;
	int blockY;

	dim3 block_size;

	dim3 num_blocks;
		
	//int2 grid_size = make_int2(this->volume_size.x, this->volume_size.y);
	//dim3 block_size(32, 8);
	//dim3 num_blocks((grid_size.x + block_size.x - 1) / block_size.x, (grid_size.y + block_size.y - 1) / block_size.y);
	switch (this->impl) {
	case 0:
		cudaOccupancyMaxPotentialBlockSize(
			&minGridSize,
			&blockSize,
			intersect_volume_kernel_single_loop,
			0,
			0
		);

		blockX = 32;
		blockY = blockSize / 32;

		if (blockY == 0) blockY = 1;

		block_size = dim3(blockX, blockY);

		num_blocks = dim3(
			(volume_size.x + block_size.x - 1) / block_size.x,
			(volume_size.y + block_size.y - 1) / block_size.y
		);
		throw_on_cuda_error();
		printf("Running implementation 0 with num_blocks: (%u, %u), block_size: (%u, %u)\n",
			num_blocks.x, num_blocks.y,
			block_size.x, block_size.y);
		intersect_volume_kernel_single_loop << <num_blocks, block_size >> > (this->volume_data_gpu_tex, this->threshold_value, this->extended_heightfield->gpu_ptr(), this->normal_map->gpu_ptr(), z_buffer.gpu_ptr(), this->volume_size, this->buffer_length, this->n_hf_entries, image_plane, false, make_int2(425, 425) );
		break;
	case 1:
		cudaOccupancyMaxPotentialBlockSize(
			&minGridSize,
			&blockSize,
			get_height_field_marching_volume_kernel,
			0,
			0
		);

		blockX = 32;
		blockY = blockSize / 32;

		if (blockY == 0) blockY = 1;

		block_size = dim3(blockX, blockY);

		num_blocks = dim3(
			(volume_size.x + block_size.x - 1) / block_size.x,
			(volume_size.y + block_size.y - 1) / block_size.y
		);
		get_height_field_marching_volume_kernel << <num_blocks, block_size >> > (this->volume_data_gpu_tex, this->threshold_value, this->extended_heightfield->gpu_ptr(), this->volume_size, this->buffer_length, this->n_hf_entries, image_plane);
		cudaOccupancyMaxPotentialBlockSize(
			&minGridSize,
			&blockSize,
			get_first_z_hit_marching_volume_kernel,
			0,
			0
		);

		blockX = 32;
		blockY = blockSize / 32;

		if (blockY == 0) blockY = 1;

		block_size = dim3(blockX, blockY);

		num_blocks = dim3(
			(volume_size.x + block_size.x - 1) / block_size.x,
			(volume_size.y + block_size.y - 1) / block_size.y
		);
		get_first_z_hit_marching_volume_kernel << <num_blocks, block_size >> > (this->volume_data_gpu_tex, this->threshold_value, z_buffer.gpu_ptr(), this->volume_size, image_plane);

		cudaOccupancyMaxPotentialBlockSize(
			&minGridSize,
			&blockSize,
			get_normal_map_marching_volume_kernel,
			0,
			0
		);

		blockX = 32;
		blockY = blockSize / 32;

		if (blockY == 0) blockY = 1;

		block_size = dim3(blockX, blockY);

		num_blocks = dim3(
			(volume_size.x + block_size.x - 1) / block_size.x,
			(volume_size.y + block_size.y - 1) / block_size.y
		);
		get_normal_map_marching_volume_kernel << <num_blocks, block_size >> > (this->volume_data_gpu_tex, this->threshold_value, this->normal_map->gpu_ptr(), z_buffer.gpu_ptr(), this->volume_size);
		break;
	case 2: 
		cudaOccupancyMaxPotentialBlockSize(
			&minGridSize,
			&blockSize,
			get_height_field_marching_volume_kernel,
			0,
			0
		);

		blockX = 32;
		blockY = blockSize / 32;

		if (blockY == 0) blockY = 1;

		block_size = dim3(blockX, blockY);

		num_blocks = dim3(
			(volume_size.x + block_size.x - 1) / block_size.x,
			(volume_size.y + block_size.y - 1) / block_size.y
		);

		get_height_field_marching_volume_kernel << <num_blocks, block_size >> > (this->volume_data_gpu_tex, this->threshold_value, this->extended_heightfield->gpu_ptr(), this->volume_size, this->buffer_length, this->n_hf_entries, image_plane);
		
		cudaOccupancyMaxPotentialBlockSize(
			&minGridSize,
			&blockSize,
			get_normal_map_single_kernel_marching_volume_kernel,
			0,
			0
		);

		blockX = 32;
		//blockY = blockSize / 32;
		blockY = 8;

		if (blockY == 0) blockY = 1;

		block_size = dim3(blockX, blockY);

		num_blocks = dim3(
			(volume_size.x + block_size.x - 1) / block_size.x,
			(volume_size.y + block_size.y - 1) / block_size.y
		);
	
		get_normal_map_single_kernel_marching_volume_kernel << <num_blocks, block_size >> > (this->volume_data_gpu_tex, this->threshold_value, this->normal_map->gpu_ptr(), z_buffer.gpu_ptr(), this->volume_size, image_plane);
		break;
	default: 
		/*
		cudaOccupancyMaxPotentialBlockSize(
			&minGridSize,
			&blockSize,
			intersect_volume_kernel,
			0,
			0
		);

		blockX = 32;
		blockY = blockSize / 32;

		if (blockY == 0) blockY = 1;

		block_size = dim3(blockX, blockY);

		num_blocks = dim3(
			(volume_size.x + block_size.x - 1) / block_size.x,
			(volume_size.y + block_size.y - 1) / block_size.y
		);*/


		auto num_blocks_and_size =  get_max_potential_block(intersect_volume_kernel, volume_size.x, volume_size.y);
		num_blocks = std::get<0>(num_blocks_and_size);
		block_size = std::get<1>(num_blocks_and_size);
		intersect_volume_kernel << <num_blocks, block_size >> > (this->volume_data_gpu_tex, this->threshold_value, this->extended_heightfield->gpu_ptr(), this->normal_map->gpu_ptr(), z_buffer.gpu_ptr(), this->volume_size, this->buffer_length, this->n_hf_entries, image_plane, false, make_int2(425, 425) );
		break;
	}


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


