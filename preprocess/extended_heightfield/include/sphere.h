#pragma once

struct Sphere
{
	float x, y, z;
	float r;
	inline bool operator()(Sphere a, Sphere b) const { return a.z + a.r < b.z + b.r; }
};
