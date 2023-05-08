#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#define empty 65535.0f
#define empty_interval make_float2( empty, empty )

void throw_on_cuda_error();

float* allocate_float_buffer_on_gpu(int3 buffer_size);
float2* allocate_float2_buffer_on_gpu(int3 buffer_size);
float3* allocate_float3_buffer_on_gpu(int3 buffer_size);