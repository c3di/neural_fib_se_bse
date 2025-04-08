#pragma once

#include "cuda_utils.h"

struct Cuboid
{
	static const size_t N_FLOAT_PARAMS = 10;
	float3 position;
	float4 orientation;
	float3 size;
	float3 aabb;
	inline bool operator()(const Cuboid& a, const Cuboid& b) const { return a.position.z + a.aabb.z < b.position.z + b.aabb.z; }
	inline Cuboid& operator=( float* data )
	{ 
		position.x = *(data++);
		position.y = *(data++);
		position.z = *(data++);
		orientation.x = *(data++); 
		orientation.y = *(data++); 
		orientation.z = *(data++); 
		orientation.w = *(data++); 
		size.x = *(data++);
		size.y = *(data++);
		size.z = *(data++);
		return *this; 
	};
} ;
