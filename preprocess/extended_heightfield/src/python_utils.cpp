#include "python_utils.h"

py::array_t<float> create_py_array(int shape0, int shape1, int shape2)
{
	return py::array(py::buffer_info(
		nullptr,                                                                   /* Pointer to data (nullptr -> ask NumPy to allocate!) */
		sizeof(float),                                                              /* Size of one item */
		py::format_descriptor<float>::value,                                        /* Buffer format */
		3,																		     /* How many dimensions? */
		{ shape0, shape1, shape2 },                                                 /* Number of elements for each dimension */
		{ shape1 * shape2 * sizeof(float), shape2 * sizeof(float), sizeof(float) }  /* Strides for each dimension */
	));
}
