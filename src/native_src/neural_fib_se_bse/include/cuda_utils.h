#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <tuple>

#define empty 65535.0f
#define empty_interval make_float2( empty, empty )

__device__ __host__ inline bool operator==(const float2& a, const float2& b) { return a.x == b.x && a.y == b.y; };
__device__ __host__ inline bool operator!=(const float2& a, const float2& b) { return a.x != b.x || a.y != b.y; };

__device__ __host__ inline bool operator==(const float3& a, const float3& b) { return a.x == b.x && a.y == b.y && a.z == b.z; };
__device__ __host__ inline bool operator!=(const float3& a, const float3& b) { return a.x != b.x || a.y != b.y || a.z != b.z; };

__device__ __host__ inline bool operator==(const int2& a, const int2& b) { return a.x == b.x && a.y == b.y; };
__device__ __host__ inline bool operator!=(const int2& a, const int2& b) { return a.x != b.x || a.y != b.y; };

__device__ __host__ inline bool operator==(const int3& a, const int3& b) { return a.x == b.x && a.y == b.y && a.z == b.z; };
__device__ __host__ inline bool operator!=(const int3& a, const int3& b) { return a.x != b.x || a.y != b.y || a.z != b.z; };

inline std::tuple<int, int> as_tuple(int2 p) { return std::tuple<int, int>(p.x, p.y); }
inline std::tuple<int, int, int> as_tuple(int3 p) { return std::tuple<int, int, int>(p.x, p.y, p.z); }

inline std::tuple<float, float> as_tuple(float2 p) { return std::tuple<float, float>(p.x, p.y); }
inline std::tuple<float, float, float> as_tuple(float3 p) { return std::tuple<float, float, float>(p.x, p.y, p.z); }

inline int2 as_int2(const std::tuple<int, int> p) { return make_int2(std::get<0>(p), std::get<1>(p)); };
inline int3 as_int3(const std::tuple<int, int, int> p) { return make_int3(std::get<0>(p), std::get<1>(p), std::get<2>(p)); };
inline float2 as_float2(const std::tuple<float, float> p) { return make_float2(std::get<0>(p), std::get<1>(p)); };
inline float3 as_float3(const std::tuple<float, float, float> p) { return make_float3(std::get<0>(p), std::get<1>(p), std::get<2>(p) ); };

inline float4 as_float4(const std::tuple<float, float, float, float> p) { return make_float4(std::get<0>(p), std::get<1>(p), std::get<2>(p), std::get<3>(p) ); };
inline float4 as_float4(const float p[4]) { return make_float4(p[0], p[1], p[2], p[3]); };

void throw_on_cuda_error();

template<typename T>
T* allocate_buffer_on_gpu(int3 buffer_size);

template<typename T>
T* allocate_buffer_on_gpu(int3 buffer_size, T init_value);

template<typename T>
void call_mem_set_kernel(T* buffer, int3 buffer_size, T init_value);

template<typename T>
inline std::tuple<dim3, dim3> get_max_potential_block(T kernel, int size_x, int size_y) {
	int min_grid_size = 0;
	int block_size_one_dim = 0;
	cudaOccupancyMaxPotentialBlockSize(
			&min_grid_size,
			&block_size_one_dim,
			kernel,
			0,
			0
		);
	int block_x = 32;
	int block_y = block_size_one_dim / block_x;

	if (block_y == 0) block_y = 1;

	dim3 block_size = dim3(block_x, block_y);

	dim3 num_blocks = dim3(
		(size_x + block_size.x - 1) / block_size.x,
		(size_y + block_size.y - 1) / block_size.y
	);

	return std::make_tuple(num_blocks, block_size);
}
