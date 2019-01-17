#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "quantum_gate.h"
#include "quantum_computer.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(qforte, m) {
    py::class_<QuantumCircuit>(m, "QuantumCircuit")
        .def(py::init<>())
        .def("add_gate", &QuantumCircuit::add_gate)
        .def("str", &QuantumCircuit::str);

    py::class_<QuantumBasis>(m, "QuantumBasis")
        .def(py::init<size_t>(), "n"_a = 0, "Make a basis element")
        .def("str", &QuantumBasis::str);

    py::class_<QuantumComputer>(m, "QuantumComputer")
        .def(py::init<size_t>(), "nqubits"_a, "Make a quantum computer with 'nqubits' qubits")
        .def("apply_circuit", &QuantumComputer::apply_circuit)
        .def("apply_gate", &QuantumComputer::apply_gate)
        .def("measure_circut", &QuantumComputer::measure_circut)
        .def("measure_gate", &QuantumComputer::measure_gate)
        .def("coeff", &QuantumComputer::coeff)
        .def("set_state", &QuantumComputer::set_state)
        .def("str", &QuantumComputer::str)
        .def("__repr__", [](const QuantumComputer& qc) {
            std::string r("QuantumComputer(\n");
            for (const std::string& s : qc.str()) {
                r += "  " + s + "\n";
            }
            r += " )";
            return r;
        });

    py::class_<QuantumGate>(m, "QuantumGate")
        .def("str", &QuantumGate::str)
        .def("__str__", &QuantumGate::str);

    m.def("make_gate", &make_gate, "type"_a, "target"_a, "control"_a, "parameter"_a = 0.0);
}
