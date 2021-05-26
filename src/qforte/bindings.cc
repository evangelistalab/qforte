#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "fmt/format.h"

#include "quantum_basis.h"
#include "quantum_circuit.h"
#include "quantum_gate.h"
#include "quantum_computer.h"
#include "quantum_operator.h"
#include "sq_operator.h"
#include "sq_op_pool.h"
#include "quantum_op_pool.h"
#include "timer.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(qforte, m) {
    py::class_<QuantumCircuit>(m, "QuantumCircuit")
        .def(py::init<>())
        .def("add", &QuantumCircuit::add_gate)
        .def("add", &QuantumCircuit::add_circuit)
        .def("add_gate", &QuantumCircuit::add_gate)
        .def("add_circuit", &QuantumCircuit::add_circuit)
        .def("gates", &QuantumCircuit::gates)
        .def("size", &QuantumCircuit::size)
        .def("adjoint", &QuantumCircuit::adjoint)
        .def("canonicalize_pauli_circuit", &QuantumCircuit::canonicalize_pauli_circuit)
        .def("set_parameters", &QuantumCircuit::set_parameters)
        .def("get_num_cnots", &QuantumCircuit::get_num_cnots)
        .def("str", &QuantumCircuit::str)
        .def("__str__", &QuantumCircuit::str)
        .def("__repr__", &QuantumCircuit::str);

    py::class_<SQOperator>(m, "SQOperator")
        .def(py::init<>())
        .def("add", &SQOperator::add_term)
        .def("add", &SQOperator::add_op)
        .def("add_term", &SQOperator::add_term)
        .def("add_op", &SQOperator::add_op)
        .def("set_coeffs", &SQOperator::set_coeffs)
        .def("terms", &SQOperator::terms)
        .def("canonical_order", &SQOperator::canonical_order)
        .def("simplify", &SQOperator::simplify)
        .def("jw_transform", &SQOperator::jw_transform)
        .def("str", &SQOperator::str)
        .def("__str__", &SQOperator::str)
        .def("__repr__", &SQOperator::str);

    py::class_<SQOpPool>(m, "SQOpPool")
        .def(py::init<>())
        .def("add", &SQOpPool::add_term)
        .def("add_term", &SQOpPool::add_term)
        .def("set_coeffs", &SQOpPool::set_coeffs)
        .def("terms", &SQOpPool::terms)
        .def("set_orb_spaces", &SQOpPool::set_orb_spaces)
        .def("get_quantum_op_pool", &SQOpPool::get_quantum_op_pool)
        .def("get_quantum_operator", &SQOpPool::get_quantum_operator, py::arg("order_type"),
             py::arg("combine_like_terms") = true)
        .def("fill_pool", &SQOpPool::fill_pool)
        .def("str", &SQOpPool::str)
        .def("__str__", &SQOpPool::str)
        .def("__repr__", &SQOpPool::str);

    py::class_<QuantumOperator>(m, "QuantumOperator")
        .def(py::init<>())
        .def("add", &QuantumOperator::add_term)
        .def("add", &QuantumOperator::add_op)
        .def("add_term", &QuantumOperator::add_term)
        .def("add_op", &QuantumOperator::add_op)
        .def("set_coeffs", &QuantumOperator::set_coeffs)
        .def("mult_coeffs", &QuantumOperator::mult_coeffs)
        .def("terms", &QuantumOperator::terms)
        .def("order_terms", &QuantumOperator::order_terms)
        .def("canonical_order", &QuantumOperator::canonical_order)
        .def("simplify", &QuantumOperator::simplify)
        .def("operator_product", &QuantumOperator::operator_product)
        .def("check_op_equivalence", &QuantumOperator::check_op_equivalence)
        .def("num_qubits", &QuantumOperator::num_qubits)
        .def("str", &QuantumOperator::str)
        .def("__str__", &QuantumOperator::str)
        .def("__repr__", &QuantumOperator::str);

    py::class_<QuantumOpPool>(m, "QuantumOpPool")
        .def(py::init<>())
        .def("add", &QuantumOpPool::add_term)
        .def("add_term", &QuantumOpPool::add_term)
        .def("set_coeffs", &QuantumOpPool::set_coeffs)
        .def("set_op_coeffs", &QuantumOpPool::set_op_coeffs)
        .def("set_terms", &QuantumOpPool::set_terms)
        .def("terms", &QuantumOpPool::terms)
        .def("join_op_from_right_lazy", &QuantumOpPool::join_op_from_right_lazy)
        .def("join_op_from_right", &QuantumOpPool::join_op_from_right)
        .def("join_op_from_left", &QuantumOpPool::join_op_from_left)
        .def("join_as_commutator", &QuantumOpPool::join_as_commutator)
        .def("square", &QuantumOpPool::square)
        .def("fill_pool", &QuantumOpPool::fill_pool)
        .def("str", &QuantumOpPool::str)
        .def("__str__", &QuantumOpPool::str)
        .def("__repr__", &QuantumOpPool::str);

    py::class_<QuantumBasis>(m, "QuantumBasis")
        .def(py::init<size_t>(), "n"_a = 0, "Make a basis element")
        .def("str", &QuantumBasis::str)
        .def("__str__", &QuantumBasis::str)
        .def("__repr__", &QuantumBasis::str)
        .def("flip_bit", &QuantumBasis::flip_bit)
        .def("set_bit", &QuantumBasis::set_bit)
        .def("add", &QuantumBasis::add)
        .def("get_bit", &QuantumBasis::get_bit);

    py::class_<Computer>(m, "Computer")
        .def(py::init<size_t>(), "nqubits"_a, "Make a quantum computer with 'nqubits' qubits")
        .def("apply_circuit_safe", &Computer::apply_circuit_safe)
        .def("apply_operator", &Computer::apply_operator)
        .def("apply_circuit", &Computer::apply_circuit)
        .def("apply_gate_safe", &Computer::apply_gate_safe)
        .def("apply_gate", &Computer::apply_gate)
        .def("apply_constant", &Computer::apply_constant)
        .def("measure_circuit", &Computer::measure_circuit)
        .def("measure_z_readouts_fast", &Computer::measure_z_readouts_fast)
        .def("measure_readouts", &Computer::measure_readouts)
        .def("perfect_measure_circuit", &Computer::perfect_measure_circuit)
        .def("direct_oppl_exp_val", &Computer::direct_oppl_exp_val)
        .def("direct_idxd_oppl_exp_val", &Computer::direct_idxd_oppl_exp_val)
        .def("direct_oppl_exp_val_w_mults", &Computer::direct_oppl_exp_val_w_mults)
        .def("direct_op_exp_val", &Computer::direct_op_exp_val)
        .def("direct_circ_exp_val", &Computer::direct_circ_exp_val)
        .def("direct_pauli_circ_exp_val", &Computer::direct_pauli_circ_exp_val)
        .def("direct_gate_exp_val", &Computer::direct_gate_exp_val)
        .def("coeff", &Computer::coeff)
        .def("get_coeff_vec", &Computer::get_coeff_vec)
        .def("get_nqubit", &Computer::get_nqubit)
        .def("set_coeff_vec", &Computer::set_coeff_vec)
        .def("set_state", &Computer::set_state)
        .def("zero_state", &Computer::zero_state)
        .def("get_timings", &Computer::get_timings)
        .def("clear_timings", &Computer::clear_timings)
        .def("str", &Computer::str)
        .def("__str__", &Computer::str)
        .def("__repr__", [](const Computer& qc) {
            std::string r("Computer(\n");
            for (const std::string& s : qc.str()) {
                r += "  " + s + "\n";
            }
            r += " )";
            return r;
        });

    py::class_<QuantumGate>(m, "QuantumGate")
        .def("target", &QuantumGate::target)
        .def("control", &QuantumGate::control)
        .def("gate_id", &QuantumGate::gate_id)
        .def("adjoint", &QuantumGate::adjoint)
        .def("str", &QuantumGate::str)
        .def("__str__", &QuantumGate::str)
        .def("__repr__", &QuantumGate::repr);

    py::class_<local_timer>(m, "local_timer")
        .def(py::init<>())
        .def("reset", &local_timer::reset)
        .def("get", &local_timer::get);

    m.def(
        "gate",
        [](std::string type, size_t target, std::complex<double> parameter) {
            // only single qubit gates accept this synthax
            auto vec = {"X",  "Y", "Z", "H", "R", "Rx",  "Ry",
                        "Rz", "V", "S", "T", "I", "Rzy", "rU1"};
            if (std::find(vec.begin(), vec.end(), type) != vec.end()) {
                return make_gate(type, target, target, parameter);
            }
            std::string msg =
                fmt::format("make_gate()\ttarget = {}, parameter = {} + {} i, is not a valid "
                            "quantum gate input for type = {}",
                            target, parameter.real(), parameter.imag(), type);
            throw std::invalid_argument(msg);
        },
        "type"_a, "target"_a, "parameter"_a = 0.0, "Make a gate.");

    m.def(
        "gate",
        [](std::string type, size_t target, size_t control) {
            // test for two-qubit gates that require no parameters
            auto vec = {"SWAP", "cV", "CNOT", "cX", "cY", "cZ"};
            if (std::find(vec.begin(), vec.end(), type) != vec.end()) {
                return make_gate(type, target, control, 0.0);
            }
            // test for one-qubit gates that require no parameters but were specified with both
            // target and control
            if (target == control) {
                auto vec2 = {
                    "X", "Y", "Z", "H", "V", "S", "T", "I", "Rzy",
                };
                if (std::find(vec2.begin(), vec2.end(), type) != vec2.end()) {
                    return make_gate(type, target, control, 0.0);
                }
            }
            std::string msg = fmt::format("make_gate()\ttarget = {}, control = {}, is not a valid "
                                          "quantum gate input for type = {}",
                                          target, control, type);
            throw std::invalid_argument(msg);
        },
        "type"_a, "target"_a, "control"_a, "Make a gate.");

    m.def(
        "gate",
        [](std::string type, size_t target, size_t control, std::complex<double> parameter) {
            return make_gate(type, target, control, parameter);
        },
        "type"_a, "target"_a, "control"_a, "parameter"_a = 0.0, "Make a gate.");

    m.def("control_gate", &make_control_gate, "control"_a, "QuantumGate"_a);
}
