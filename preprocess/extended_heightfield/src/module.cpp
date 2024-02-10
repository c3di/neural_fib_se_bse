#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

#include "HeightfieldExtractor.h"
#include "CSG_Resolver.h"
#include "Sphere_Intersector.h"
#include "Cylinder_Intersector.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<float>);
PYBIND11_MAKE_OPAQUE(std::vector<float2>);
PYBIND11_MAKE_OPAQUE(std::vector<float3>);
PYBIND11_MAKE_OPAQUE(std::vector<int2>);
PYBIND11_MAKE_OPAQUE(std::vector<int3>);

PYBIND11_MODULE(extended_heightfield, m) 
{
    // CUDA vector types
    PYBIND11_NUMPY_DTYPE(float2, x, y);
    PYBIND11_NUMPY_DTYPE(float3, x, y, z);

    // rasterizer API
    py::class_<Sphere_Intersector>(m, "Sphere_Rasterizer")
        .def(py::init<std::tuple<int, int>, int, int>())
        .def("intersect", &Sphere_Intersector::intersect_py, py::arg("image_plane"));

    py::class_<Cylinder_Intersector>(m, "Cylinder_Rasterizer")
        .def(py::init<std::tuple<int, int>, int, int>())
        .def("intersect", &Cylinder_Intersector::intersect_py, py::arg("image_plane"))
        .def("get_extended_height_field", &Cylinder_Intersector::get_extended_height_field_py);

    py::class_<HeightFieldExtractor>(m, "HeightFieldExtractor")
        .def(py::init<std::tuple<int, int>, int, int>())
        .def("extract_data_representation", &HeightFieldExtractor::extract_data_representation_py, py::arg("image_plane"))
        .def("add_spheres",   &HeightFieldExtractor::add_spheres_py,   py::arg("spheres"))
        .def("add_cylinders", &HeightFieldExtractor::add_cylinders_py, py::arg("cylinders"))
        .def("add_cuboids",   &HeightFieldExtractor::add_cuboids_py, py::arg("cuboids"));

    py::class_<CSG_Resolver>(m, "CSG_Resolver")
        .def(py::init<py::array&, int>())
        .def("resolve_csg", &CSG_Resolver::resolve_csg_py, py::arg("image_plane"));

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
