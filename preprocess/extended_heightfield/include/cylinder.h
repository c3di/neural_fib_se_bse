#pragma once

struct Cylinder
{
	static const size_t N_FLOAT_PARAMS = 8;
	float x, y, z;
	float3 orientation;
	float r, l;
	float sx, sy, sz;
	inline bool operator()(const Cylinder& a, const Cylinder& b) const { return a.z + a.r < b.z + b.r; }
} ;
