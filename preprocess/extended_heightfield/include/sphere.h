#pragma once

#include "cuda_utils.h"

struct Sphere
{
	static const size_t N_FLOAT_PARAMS = 4;
	float x, y, z;
	float r;
	inline bool operator()(const Sphere& a, const Sphere& b) const { return a.z + a.r < b.z + b.r; }
	inline Sphere& operator=(float* data)
	{
		x = *(data++);
		y = *(data++);
		z = *(data++);
		r = *(data++);
		return *this;
	};

};
