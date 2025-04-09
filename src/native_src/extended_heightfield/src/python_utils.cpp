#include "python_utils.h"

py::array_t<float> create_py_array(int shape0, int shape1, int shape2)
{
	size_t size = shape0 * shape1 * shape2;
	float* data = new float[size];

	py::capsule free_when_done( data, [](void* f) {
		double* foo = reinterpret_cast<double*>(f);
		delete[] foo;
	} );

	return py::array_t<float>(
		{ shape0, shape1, shape2 },                                                  // Number of elements for each dimension
		{ shape1 * shape2 * sizeof(float), shape2 * sizeof(float), sizeof(float) },  // Strides for each dimension
		data,
		free_when_done );
}
