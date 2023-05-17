#pragma once

#include "cuda_utils.h"

struct Sphere
{
	static const size_t N_FLOAT_PARAMS = 4;
	float3 position;
	float r;
	inline bool operator()(const Sphere& a, const Sphere& b) const { return a.position.z + a.r < b.position.z + b.r; }
	inline Sphere& operator=(float* data)
	{
		position.x = *(data++);
		position.y = *(data++);
		position.z = *(data++);
		r = *(data++);
		return *this;
	};

};
