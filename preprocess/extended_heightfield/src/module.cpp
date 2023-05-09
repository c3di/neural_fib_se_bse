#include <pybind11/pybind11.h>

#include "HeightFieldExtractor.h"
#include "CSG_Resolver.h"
#include "Sphere_Rasterizer.h"

namespace py = pybind11;

PYBIND11_MODULE(extended_heightfield, m) 
{
    // inpainting API
    py::class_<Sphere_Rasterizer>(m, "Sphere_Rasterizer")
        .def(py::init<py::array&, std::pair<int, int>, int, int>())
        .def("rasterize_spheres", &Sphere_Rasterizer::rasterize_py, py::arg("image_plane"));

    py::class_<HeightFieldExtractor>(m, "HeightFieldExtractor")
        .def(py::init<py::array&, std::pair<int, int>, int, int>())
        .def("extract_data_representation", &HeightFieldExtractor::extract_data_representation_py, py::arg("image_plane"));

    py::class_<CSG_Resolver>(m, "CSG_Resolver")
        .def(py::init<py::array&, int>())
        .def("resolve_csg", &CSG_Resolver::resolve_csg_py, py::arg("image_plane"));

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
