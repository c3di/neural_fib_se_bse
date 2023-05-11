#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#define empty 65535.0f
#define empty_interval make_float2( empty, empty )

__device__ inline bool operator!=(const float2& a, const float2& b) { return a.x == b.x && a.y == b.y; };
__device__ inline bool operator==(const float2& a, const float2& b) { return a.x != b.x || a.y != b.y; };

void throw_on_cuda_error();

template<typename T>
T* allocate_buffer_on_gpu(int3 buffer_size);

template<typename T>
T* allocate_buffer_on_gpu(int3 buffer_size, T init_value);