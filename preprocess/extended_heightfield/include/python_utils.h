#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <cuda.h>
#include <cuda_runtime.h>

py::array_t<float> create_py_array(int shape0, int shape1, int shape2);

inline std::pair<int, int> tuple( int2 p ) { return std::pair<int, int>( p.x, p.y); }