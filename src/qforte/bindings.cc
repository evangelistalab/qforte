#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

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
        .def("add_gate", &QuantumCircuit::add_gate)
        .def("add_circuit", &QuantumCircuit::add_circuit)
        .def("gates", &QuantumCircuit::gates)
        .def("size", &QuantumCircuit::size)
        .def("adjoint", &QuantumCircuit::adjoint)
        .def("canonical_order", &QuantumCircuit::canonical_order)
        .def("set_parameters", &QuantumCircuit::set_parameters)
        .def("get_num_cnots", &QuantumCircuit::get_num_cnots)
        .def("str", &QuantumCircuit::str);

    py::class_<SQOperator>(m, "SQOperator")
        .def(py::init<>())
        .def("add_term", &SQOperator::add_term)
        .def("add_op", &SQOperator::add_op)
        .def("set_coeffs", &SQOperator::set_coeffs)
        .def("terms", &SQOperator::terms)
        .def("canonical_order", &SQOperator::canonical_order)
        .def("simplify", &SQOperator::simplify)
        .def("jw_transform", &SQOperator::jw_transform)
        .def("str", &SQOperator::str);

    py::class_<SQOpPool>(m, "SQOpPool")
        .def(py::init<>())
        .def("add_term", &SQOpPool::add_term)
        .def("set_coeffs", &SQOpPool::set_coeffs)
        .def("terms", &SQOpPool::terms)
        .def("set_orb_spaces", &SQOpPool::set_orb_spaces)
        .def("get_quantum_operators", &SQOpPool::get_quantum_operators)
        .def("get_quantum_op_pool", &SQOpPool::get_quantum_op_pool)
        .def("get_quantum_operator", &SQOpPool::get_quantum_operator)
        .def("fill_pool", &SQOpPool::fill_pool)
        .def("str", &SQOpPool::str);

    py::class_<QuantumOperator>(m, "QuantumOperator")
        .def(py::init<>())
        .def("add_term", &QuantumOperator::add_term)
        .def("add_op", &QuantumOperator::add_op)
        .def("set_coeffs", &QuantumOperator::set_coeffs)
        .def("terms", &QuantumOperator::terms)
        .def("order_terms", &QuantumOperator::order_terms)
        .def("canonical_order", &QuantumOperator::canonical_order)
        .def("simplify", &QuantumOperator::simplify)
        .def("join_operator", &QuantumOperator::join_operator)
        .def("join_operator_lazy", &QuantumOperator::join_operator_lazy)
        .def("check_op_equivalence", &QuantumOperator::check_op_equivalence)
        .def("str", &QuantumOperator::str);

    py::class_<QuantumOpPool>(m, "QuantumOpPool")
        .def(py::init<>())
        .def("add_term", &QuantumOpPool::add_term)
        .def("set_coeffs", &QuantumOpPool::set_coeffs)
        .def("set_op_coeffs", &QuantumOpPool::set_op_coeffs)
        .def("set_terms", &QuantumOpPool::set_terms)
        .def("terms", &QuantumOpPool::terms)
        .def("set_orb_spaces", &QuantumOpPool::set_orb_spaces)
        .def("join_op_from_right_lazy", &QuantumOpPool::join_op_from_right_lazy)
        .def("join_op_from_right", &QuantumOpPool::join_op_from_right)
        .def("join_op_from_left", &QuantumOpPool::join_op_from_left)
        .def("join_as_comutator", &QuantumOpPool::join_as_comutator)
        .def("square", &QuantumOpPool::square)
        .def("fill_pool", &QuantumOpPool::fill_pool)
        .def("str", &QuantumOpPool::str);

    py::class_<QuantumBasis>(m, "QuantumBasis")
        .def(py::init<size_t>(), "n"_a = 0, "Make a basis element")
        .def("str", &QuantumBasis::str)
        .def("flip_bit", &QuantumBasis::flip_bit)
        .def("get_bit", &QuantumBasis::get_bit);

    py::class_<QuantumComputer>(m, "QuantumComputer")
        .def(py::init<size_t>(), "nqubits"_a, "Make a quantum computer with 'nqubits' qubits")
        .def("apply_circuit_safe", &QuantumComputer::apply_circuit_safe)
        .def("apply_operator", &QuantumComputer::apply_operator)
        .def("apply_circuit", &QuantumComputer::apply_circuit)
        .def("apply_gate_safe", &QuantumComputer::apply_gate_safe)
        .def("apply_gate", &QuantumComputer::apply_gate)
        .def("apply_constant",  &QuantumComputer::apply_constant)
        .def("measure_circuit", &QuantumComputer::measure_circuit)
        .def("measure_z_readouts_fast", &QuantumComputer::measure_z_readouts_fast)
        .def("measure_readouts", &QuantumComputer::measure_readouts)
        .def("perfect_measure_circuit", &QuantumComputer::perfect_measure_circuit)
        .def("direct_oppl_exp_val", &QuantumComputer::direct_oppl_exp_val)
        .def("direct_idxd_oppl_exp_val", &QuantumComputer::direct_idxd_oppl_exp_val)
        .def("direct_oppl_exp_val_w_mults", &QuantumComputer::direct_oppl_exp_val_w_mults)
        .def("direct_op_exp_val", &QuantumComputer::direct_op_exp_val)
        .def("direct_circ_exp_val", &QuantumComputer::direct_circ_exp_val)
        .def("direct_pauli_circ_exp_val", &QuantumComputer::direct_pauli_circ_exp_val)
        .def("direct_gate_exp_val", &QuantumComputer::direct_gate_exp_val)
        .def("coeff", &QuantumComputer::coeff)
        .def("get_coeff_vec", &QuantumComputer::get_coeff_vec)
        .def("get_nqubit", &QuantumComputer::get_nqubit)
        .def("set_coeff_vec", &QuantumComputer::set_coeff_vec)
        .def("set_state", &QuantumComputer::set_state)
        .def("zero_state", &QuantumComputer::zero_state)
        .def("get_timings", &QuantumComputer::get_timings)
        .def("clear_timings", &QuantumComputer::clear_timings)
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
        .def("adjoint", &QuantumGate::adjoint)
        .def("__str__", &QuantumGate::str)
        .def("__repr__", &QuantumGate::repr);

    py::class_<local_timer>(m, "local_timer")
        .def(py::init<>())
        .def("reset", &local_timer::reset)
        .def("get", &local_timer::get);

    m.def("make_gate", &make_gate, "type"_a, "target"_a, "control"_a, "parameter"_a = 0.0);
    m.def("make_control_gate", &make_control_gate, "control"_a, "QuantumGate"_a);
}
