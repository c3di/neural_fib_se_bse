#include <pybind11/pybind11.h>

#include "HeightFieldExtractor.h"
#include "CSG_Resolver.h"
#include "Sphere_Rasterizer.h"
#include "Cylinder_Rasterizer.h"

namespace py = pybind11;

PYBIND11_MODULE(extended_heightfield, m) 
{
    // rasterizer API
    py::class_<Sphere_Rasterizer>(m, "Sphere_Rasterizer")
        .def(py::init<std::pair<int, int>, int, int>())
        .def("rasterize", &Sphere_Rasterizer::rasterize_py, py::arg("image_plane"));

    py::class_<Cylinder_Rasterizer>(m, "Cylinder_Rasterizer")
        .def(py::init<std::pair<int, int>, int, int>())
        .def("rasterize", &Cylinder_Rasterizer::rasterize_py, py::arg("image_plane"))
        .def("get_extended_height_field", &Cylinder_Rasterizer::get_extended_height_field_py);

    py::class_<HeightFieldExtractor>(m, "HeightFieldExtractor")
        .def(py::init<std::pair<int, int>, int, int>())
        .def("extract_data_representation", &HeightFieldExtractor::extract_data_representation_py, py::arg("image_plane"))
        .def("add_spheres",   &HeightFieldExtractor::add_spheres_py,   py::arg("spheres"))
        .def("add_cylinders", &HeightFieldExtractor::add_cylinders_py, py::arg("cylinders"));

    py::class_<CSG_Resolver>(m, "CSG_Resolver")
        .def(py::init<py::array&, int>())
        .def("resolve_csg", &CSG_Resolver::resolve_csg_py, py::arg("image_plane"));

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
