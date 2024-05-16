#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

#include "fmt/format.h"

#include "find_irrep.h"
#include "qubit_basis.h"
#include "circuit.h"
#include "gate.h"
#include "computer.h"
#include "qubit_operator.h"
#include "sq_operator.h"
#include "sq_op_pool.h"
#include "qubit_op_pool.h"
#include "sparse_tensor.h"
#include "timer.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(qforte, m) {
    py::class_<Circuit, std::shared_ptr<Circuit>>(m, "Circuit")
        .def(py::init<>())
        .def(py::init<const Circuit&>())
        .def("add", &Circuit::add_gate)
        .def("add", &Circuit::add_circuit)
        .def("add_gate", &Circuit::add_gate)
        .def("add_circuit", &Circuit::add_circuit)
        .def("insert_gate", &Circuit::insert_gate)
        .def("remove_gate", &Circuit::remove_gate, "pos"_a, "Remove a gate at position pos")
        .def("swap_gates", &Circuit::swap_gates, "pos1"_a, "pos2"_a,
             "Swap the gates at positions pos1 and pos2")
        .def("insert_circuit", &Circuit::insert_circuit, "pos"_a, "circ"_a,
             "Insert a circuit at position pos")
        .def("remove_gates", &Circuit::remove_gates, "pos1"_a, "pos2"_a,
             "Remove the gates in the range of positions [pos1,pos2)")
        .def("replace_gate", &Circuit::replace_gate, "pos"_a, "gate"_a,
             "Replace the gate at position pos with the given gate")
        .def("gates", &Circuit::gates)
        .def("gate", [](const Circuit& circ, size_t pos) { return circ.gates()[pos]; })
        .def("sparse_matrix", &Circuit::sparse_matrix)
        .def("size", &Circuit::size)
        .def("adjoint", &Circuit::adjoint)
        .def("canonicalize_pauli_circuit", &Circuit::canonicalize_pauli_circuit)
        .def("set_parameters", &Circuit::set_parameters)
        .def("set_parameter", &Circuit::set_parameter)
        .def("get_parameters", &Circuit::get_parameters)
        .def("get_num_cnots", &Circuit::get_num_cnots)
        .def("is_pauli", &Circuit::is_pauli)
        .def("simplify", &Circuit::simplify)
        .def("str", &Circuit::str)
        .def("__str__", &Circuit::str)
        .def("__repr__", &Circuit::str)
        .def("__eq__", [](const Circuit& a, const Circuit& b) { return a == b; });

    py::class_<SQOperator>(m, "SQOperator")
        .def(py::init<>())
        .def("add", &SQOperator::add_term)
        .def("add", &SQOperator::add_op)
        .def("add_term", &SQOperator::add_term)
        .def("add_op", &SQOperator::add_op)
        .def("set_coeffs", &SQOperator::set_coeffs)
        .def("mult_coeffs", &SQOperator::mult_coeffs)
        .def("terms", &SQOperator::terms)
        .def("canonical_order", &SQOperator::canonical_order)
        .def("simplify", &SQOperator::simplify)
        .def("jw_transform", &SQOperator::jw_transform, py::arg("qubit_excitation") = false)
        .def("str", &SQOperator::str)
        .def("__str__", &SQOperator::str)
        .def("__repr__", &SQOperator::str);

    py::class_<SQOpPool>(m, "SQOpPool")
        .def(py::init<>())
        .def("add", &SQOpPool::add_term)
        .def("add_term", &SQOpPool::add_term)
        .def("set_coeffs", &SQOpPool::set_coeffs)
        .def("terms", &SQOpPool::terms)
        .def("set_orb_spaces", &SQOpPool::set_orb_spaces, py::arg("ref"),
             py::arg("orb_irreps_to_int") = std::vector<size_t>{})
        .def("get_qubit_op_pool", &SQOpPool::get_qubit_op_pool)
        .def("get_qubit_operator", &SQOpPool::get_qubit_operator, py::arg("order_type"),
             py::arg("combine_like_terms") = true, py::arg("qubit_excitations") = false)
        .def("fill_pool", &SQOpPool::fill_pool)
        .def("str", &SQOpPool::str)
        .def("__getitem__", [](const SQOpPool& pool, size_t i) { return pool.terms()[i]; })
        .def(
            "__iter__", [](const SQOpPool& pool) { return py::make_iterator(pool.terms()); },
            py::keep_alive<0, 1>())
        .def("__len__", [](const SQOpPool& pool) { return pool.terms().size(); })
        .def("__str__", &SQOpPool::str)
        .def("__repr__", &SQOpPool::str);

    py::class_<QubitOperator>(m, "QubitOperator")
        .def(py::init<>())
        .def("add", &QubitOperator::add_term)
        .def("add", &QubitOperator::add_op)
        .def("add_term", &QubitOperator::add_term)
        .def("add_op", &QubitOperator::add_op)
        .def("set_coeffs", &QubitOperator::set_coeffs)
        .def("mult_coeffs", &QubitOperator::mult_coeffs)
        .def("terms", &QubitOperator::terms)
        .def("order_terms", &QubitOperator::order_terms)
        .def("canonical_order", &QubitOperator::canonical_order)
        .def("simplify", &QubitOperator::simplify)
        .def("operator_product", &QubitOperator::operator_product)
        .def("check_op_equivalence", &QubitOperator::check_op_equivalence)
        .def("num_qubits", &QubitOperator::num_qubits)
        .def("sparse_matrix", &QubitOperator::sparse_matrix)
        .def("str", &QubitOperator::str)
        .def(
            "__iter__", [](const QubitOperator& op) { return py::make_iterator(op.terms()); },
            py::keep_alive<0, 1>())
        .def("__str__", &QubitOperator::str)
        .def("__repr__", &QubitOperator::str);

    py::class_<QubitOpPool>(m, "QubitOpPool")
        .def(py::init<>())
        .def("add", &QubitOpPool::add_term)
        .def("add_term", &QubitOpPool::add_term)
        .def("set_coeffs", &QubitOpPool::set_coeffs)
        .def("set_op_coeffs", &QubitOpPool::set_op_coeffs)
        .def("set_terms", &QubitOpPool::set_terms)
        .def("terms", &QubitOpPool::terms)
        .def("join_op_from_right_lazy", &QubitOpPool::join_op_from_right_lazy)
        .def("join_op_from_right", &QubitOpPool::join_op_from_right)
        .def("join_op_from_left", &QubitOpPool::join_op_from_left)
        .def("join_as_commutator", &QubitOpPool::join_as_commutator)
        .def("square", &QubitOpPool::square)
        .def("fill_pool", &QubitOpPool::fill_pool)
        .def("str", &QubitOpPool::str)
        .def(
            "__iter__", [](const QubitOpPool& pool) { return py::make_iterator(pool.terms()); },
            py::keep_alive<0, 1>())
        .def("__str__", &QubitOpPool::str)
        .def("__repr__", &QubitOpPool::str);

    py::class_<QubitBasis>(m, "QubitBasis")
        .def(py::init<size_t>(), "n"_a = 0, "Make a basis element")
        .def("str", &QubitBasis::str)
        .def("__str__", &QubitBasis::default_str)
        .def("__repr__", &QubitBasis::default_str)
        .def("flip_bit", &QubitBasis::flip_bit)
        .def("set_bit", &QubitBasis::set_bit)
        .def("index", &QubitBasis::index)
        .def("get_bit", &QubitBasis::get_bit);

    py::class_<Computer, std::shared_ptr<Computer>>(m, "Computer")
        .def(py::init<size_t, double>(), "nqubits"_a, "print_threshold"_a = 1.0e-6,
             "Make a quantum computer with 'nqubits' qubits")
        .def(py::init<const Computer&>())
        .def("apply_circuit_safe", &Computer::apply_circuit_safe)
        .def("apply_matrix", &Computer::apply_matrix)
        .def("apply_sparse_matrix", &Computer::apply_sparse_matrix)
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
        .def("set_coeff_vec_from_numpy",
             [](Computer& computer, py::array_t<std::complex<double>> np_array) {
                 py::buffer_info buf_info = np_array.request();
                 std::vector<double_c> c_vec(static_cast<double_c*>(buf_info.ptr),
                                             static_cast<double_c*>(buf_info.ptr) + buf_info.size);
                 computer.set_coeff_vec(c_vec);
             })
        .def("set_state", &Computer::set_state)
        .def("null_state", &Computer::null_state)
        .def("reset", &Computer::reset, "Reset the quantum computer to the state |0>")
        .def("get_timings", &Computer::get_timings)
        .def("clear_timings", &Computer::clear_timings)
        .def("str", &Computer::str)
        .def("__str__", &Computer::str)
        .def("__repr__", &Computer::str)
        .def("__eq__", [](const Computer& a, const Computer& b) { return a == b; });

    py::class_<Gate>(m, "Gate")
        .def("gate_id", &Gate::gate_id)
        .def("target", &Gate::target)
        .def("control", &Gate::control)
        .def("sparse_matrix", &Gate::sparse_matrix)
        .def("adjoint", &Gate::adjoint)
        .def("nqubits", &Gate::nqubits)
        .def("str", &Gate::str)
        .def("has_parameter", &Gate::has_parameter)
        .def("parameter", &Gate::parameter,
             "Return the parameter associated with the gate. Returns None if the gate does not "
             "have a parameter")
        .def("__str__", &Gate::str)
        .def("__repr__", &Gate::repr)
        .def("__eq__", [](const Gate& a, const Gate& b) { return a == b; })
        .def("__neq__", [](const Gate& a, const Gate& b) { return a != b; })
        .def(
            "update_parameter",
            [](Gate& gate, double parameter) {
                if (!gate.has_parameter()) {
                    throw std::invalid_argument("Gate does not have a parameter.");
                }
                return make_gate(gate.gate_id(), gate.target(), gate.control(), parameter);
            },
            "Return a new gate with the same target, control, and gate_id, but with the given "
            "parameter.");

    py::class_<SparseVector>(m, "SparseVector")
        .def(py::init<>())
        .def("get_element", &SparseVector::get_element)
        .def("set_element", &SparseVector::set_element)
        .def("to_map", &SparseVector::to_map);

    py::class_<SparseMatrix>(m, "SparseMatrix")
        .def(py::init<>())
        .def("get_element", &SparseMatrix::get_element)
        .def("set_element", &SparseMatrix::set_element)
        .def("to_vec_map", &SparseMatrix::to_vec_map)
        .def("to_map", &SparseMatrix::to_map);

    py::class_<local_timer>(m, "local_timer")
        .def(py::init<>())
        .def("reset", &local_timer::reset)
        .def("get", &local_timer::get);

    m.def(
        "gate",
        [](std::string type, size_t target, std::complex<double> parameter) {
            if (parameter.imag() != 0.0) {
                throw std::invalid_argument("Gate parameter must be real.");
            }

            // only single qubit gates accept this synthax
            auto vec = {"X", "Y", "Z", "H", "R", "Rx", "Ry", "Rz", "V", "S", "T", "I"};
            if (std::find(vec.begin(), vec.end(), type) != vec.end()) {
                return make_gate(type, target, target, parameter.real());
            }
            std::string msg = fmt::format("make_gate()\ttarget = {}, parameter = {} is not a valid "
                                          "quantum gate input for type = {}",
                                          target, parameter.real(), type);
            throw std::invalid_argument(msg);
        },
        "type"_a, "target"_a, "parameter"_a = 0.0, "Make a gate.");

    m.def(
        "gate",
        [](std::string type, size_t target, size_t control) {
            // test for two-qubit gates that require no parameters
            auto vec = {"SWAP", "cV", "CNOT", "cX", "aCNOT", "acX", "cY", "cZ"};
            if (std::find(vec.begin(), vec.end(), type) != vec.end()) {
                return make_gate(type, target, control, 0.0);
            }
            // test for one-qubit gates that require no parameters but were specified with both
            // target and control
            if (target == control) {
                auto vec2 = {
                    "X", "Y", "Z", "H", "V", "S", "T", "I",
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
            if (parameter.imag() != 0.0) {
                throw std::invalid_argument("Gate parameter must be real.");
            }
            return make_gate(type, target, control, parameter.real());
        },
        "type"_a, "target"_a, "control"_a, "parameter"_a = 0.0, "Make a gate.");

    m.def("control_gate", &make_control_gate, "control"_a, "Gate"_a);

    m.def("evaluate_gate_interaction", &evaluate_gate_interaction, "Gate"_a, "Gate"_a);

    m.def(
        "prepare_computer_from_circuit",
        [](int nqubit, const Circuit& circuit) {
            auto computer = Computer(nqubit);
            computer.apply_circuit(circuit);
            return computer;
        },
        "nqubit"_a, "gates"_a, "A convenience function to prepare a state from a list of gates.");

    m.def(
        "prepare_computer_from_gates",
        [](int nqubits, const std::vector<Gate>& gates) {
            auto computer = Computer(nqubits);
            for (const auto& gate : gates) {
                computer.apply_gate(gate);
            }
            return computer;
        },
        "nqubits"_a, "gates"_a, "A convenience function to prepare a state from a list of gates.");

    m.def(
        "add_gate_to_computer",
        [](const Gate& gate, Computer& computer) {
            auto new_computer = Computer(computer);
            new_computer.apply_gate(gate);
            return new_computer;
        },
        "gate"_a, "computer"_a, "Return a new computer with the given gate applied to it.");

    m.def("inner_product", &dot, "a"_a, "b"_a,
          "Return the inner product of the states stored in two quantum computers.");
    m.def(
        "find_irrep",
        [](const std::vector<size_t>& orb_irrep_to_int,
           const std::vector<size_t>& spinorb_indices) -> size_t {
            /*
             * Find the irrep of a given set of spinorbitals.
             *
             * @param orb_irrep_to_int: List of integers where the i-th element is the irrep of
             * spatial orbital i.
             * @param spinorb_indices: List of spinorbital indices.
             * @return Integer representing the irrep (in Cotton ordering) of the given set of
             * spinorbitals.
             */
            return find_irrep(orb_irrep_to_int, spinorb_indices);
        },
        R"pbdoc(
               Function that finds the irreducible representation of a given set of spinorbitals.
               
               :param orb_irrep_to_int: List of integers where the i-th element is the irrep of spatial orbital i.
               :param spinorb_indices: List of spinorbital indices.
               :return: Integer representing the irrep (in Cotton ordering) of the given set of spinorbitals.
           )pbdoc",
        py::arg("orb_irrep_to_int"), py::arg("spinorb_indices"));
}
