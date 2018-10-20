#include <pybind11/pybind11.h>
#include "fn.h"

namespace py = pybind11;

PYBIND11_MODULE(qforte, m)
{
    m.def("add", &add);
    m.def("subtract", &subtract);
}
