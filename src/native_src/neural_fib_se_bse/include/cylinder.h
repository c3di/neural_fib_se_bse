#pragma once

#include "cuda_utils.h"

struct Cylinder
{
	float4 orientation;
	float3 position;
	float r, l;
	float3 aabb;
	static const size_t N_FLOAT_PARAMS = 9;

	inline bool operator()(const Cylinder& a, const Cylinder& b) const { return a.position.z + a.r < b.position.z + b.r; }
	inline Cylinder& operator=( float* data ) 
	{ 
		position.x = *(data++);
		position.y = *(data++);
		position.z = *(data++);
		orientation.x = *(data++); 
		orientation.y = *(data++); 
		orientation.z = *(data++); 
		orientation.w = *(data++); 
		r = *(data++); 
		l = *(data++); 
		return *this; 
	};
} ;
