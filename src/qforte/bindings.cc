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
        .def("gates", &QuantumCircuit::gates)
        .def("reversed_gates", &QuantumCircuit::reversed_gates)
        .def("set_parameters", &QuantumCircuit::set_parameters)
        .def("set_reversed_gates", &QuantumCircuit::set_reversed_gates)
        .def("str", &QuantumCircuit::str);

    py::class_<QuantumOperator>(m, "QuantumOperator")
        .def(py::init<bool>(), "is_mirror"_a = false, "make a quantum operator")
        .def("add_term", &QuantumOperator::add_term)
        .def("get_is_mirror", &QuantumOperator::get_is_mirror)
        .def("terms", &QuantumOperator::terms);
        //.def("str", &QuantumOperator::str);

    py::class_<QuantumBasis>(m, "QuantumBasis")
        .def(py::init<size_t>(), "n"_a = 0, "Make a basis element")
        .def("str", &QuantumBasis::str);

    py::class_<QuantumComputer>(m, "QuantumComputer")
        .def(py::init<size_t>(), "nqubits"_a, "Make a quantum computer with 'nqubits' qubits")
        .def("apply_circuit", &QuantumComputer::apply_circuit)
        .def("apply_gate", &QuantumComputer::apply_gate)
        .def("measure_circuit", &QuantumComputer::measure_circuit)
        .def("measure_rotated_circuit", &QuantumComputer::measure_rotated_circuit)
        .def("direct_op_exp_val", &QuantumComputer::direct_op_exp_val)
        .def("direct_circ_exp_val", &QuantumComputer::direct_circ_exp_val)
        .def("direct_gate_exp_val", &QuantumComputer::direct_gate_exp_val)
        .def("coeff", &QuantumComputer::coeff)
        .def("get_coeff_vec", &QuantumComputer::get_coeff_vec)
        .def("set_state", &QuantumComputer::set_state)
        .def("zero_state", &QuantumComputer::zero_state)
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
        .def("target", &QuantumGate::target)
        .def("control", &QuantumGate::control)
        .def("gate_id", &QuantumGate::gate_id)
        .def("__str__", &QuantumGate::str);

    m.def("make_gate", &make_gate, "type"_a, "target"_a, "control"_a, "parameter"_a = 0.0, "mirror"_a = false);
}
