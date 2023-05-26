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