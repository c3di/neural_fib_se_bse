#include <pybind11/pybind11.h>

#include "Sphere_Rasterizer.h"

namespace py = pybind11;

PYBIND11_MODULE(extended_heightfield, m) 
{
    // inpainting API
    py::class_<Sphere_Rasterizer>(m, "Sphere_Rasterizer")
        .def(py::init<py::array&, std::pair<int,int>, int>())
        .def("rasterize_spheres", &Sphere_Rasterizer::rasterize_spheres_py, py::arg("image_plane"));

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}