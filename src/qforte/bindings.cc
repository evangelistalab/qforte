#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "fmt/format.h"

#include "helpers.h"
#include "qubit_basis.h"
#include "qubit_basis_vector.h"
#include "circuit.h"
#include "gate.h"
#include "computer.h"
#include "pauli_string.h"
#include "pauli_string_vector.h"
#include "qubit_operator.h"
#include "sq_operator.h"
#include "sq_op_pool.h"
#include "qubit_op_pool.h"
#include "sparse_tensor.h"
#include "timer.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(qforte, m) {
    py::class_<Circuit>(m, "Circuit")
        .def(py::init<>())
        .def("add", &Circuit::add_gate)
        .def("add", &Circuit::add_circuit)
        .def("add_gate", &Circuit::add_gate)
        .def("add_circuit", &Circuit::add_circuit)
        .def("gates", &Circuit::gates)
        .def("sparse_matrix", &Circuit::sparse_matrix)
        .def("size", &Circuit::size)
        .def("adjoint", &Circuit::adjoint)
        .def("canonicalize_pauli_circuit", &Circuit::canonicalize_pauli_circuit)
        .def("set_parameters", &Circuit::set_parameters)
        .def("get_num_cnots", &Circuit::get_num_cnots)
        .def("str", &Circuit::str)
        .def("__str__", &Circuit::str)
        .def("__repr__", &Circuit::str);

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
        .def("get_qubit_op_pool", &SQOpPool::get_qubit_op_pool)
        .def("get_qubit_operator", &SQOpPool::get_qubit_operator, py::arg("order_type"),
             py::arg("combine_like_terms") = true)
        .def("fill_pool", &SQOpPool::fill_pool)
        .def("str", &SQOpPool::str)
        .def("__getitem__", [](const SQOpPool &pool, size_t i) { return pool.terms()[i]; })
        .def("__iter__", [](const SQOpPool &pool) { return py::make_iterator(pool.terms()); },
            py::keep_alive<0, 1>())
        .def("__len__", [](const SQOpPool &pool) { return pool.terms().size(); })
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
        .def("__str__", &QubitOpPool::str)
        .def("__repr__", &QubitOpPool::str);

    py::class_<QubitBasis>(m, "QubitBasis")
        .def(py::init<size_t, std::complex<double>>(), "n"_a = 0, "coeff"_a = 1.0, "Make a basis element")
        .def("str", &QubitBasis::str)
        .def("__str__", [](const QubitBasis& qb) {
            return qb.str(QubitBasis::max_qubits());
        })
        .def("__repr__", [](const QubitBasis& qb) {
            return qb.str(QubitBasis::max_qubits());
        })
        .def("__mul__",[](const QubitBasis& rhs, const std::complex<double> lhs){return multiply(lhs,rhs);}, py::is_operator())
        .def("__rmul__",[](const QubitBasis& rhs, const std::complex<double> lhs){return multiply(lhs,rhs);}, py::is_operator())
        .def("__rmul__",[](const QubitBasis& rhs, const PauliString& lhs){return apply(lhs,rhs);}, py::is_operator())
        .def("__rmul__",[](const QubitBasis& rhs, const PauliStringVector lhs){return apply(lhs,rhs);}, py::is_operator())
        .def("__add__",[](const QubitBasis& lhs, const QubitBasis& rhs){return add(lhs,rhs);}, py::is_operator())
        .def("__sub__",[](const QubitBasis& lhs, const QubitBasis& rhs){return subtract(lhs,rhs);}, py::is_operator())
        .def("flip_bit", &QubitBasis::flip_bit)
        .def("set_bit", &QubitBasis::set_bit)
        .def("add", &QubitBasis::address)
        .def("address", &QubitBasis::address)
        .def("get_bit", &QubitBasis::get_bit);
        
    py::class_<QubitBasisVector>(m, "QubitBasisVector")
        .def(py::init<std::vector<QubitBasis>>(), "Make a vector of Pauli string")
        .def("str", &QubitBasisVector::str)
        .def("__str__", [](const QubitBasisVector& vec){return join(vec.str(), "\n");})
        .def("__repr__", [](const QubitBasisVector& vec){return join(vec.str(), "\n");})
        .def("__getitem__", [](const QubitBasisVector& vec, unsigned int i){return vec[i];})
        .def("__len__", [](const QubitBasisVector& vec){return vec.get_vec().size();})
        .def("__iter__", [](const QubitBasisVector& QBasisVector){
                std::vector<QubitBasis> vec(QBasisVector.get_vec());
                return py::make_iterator(vec.begin(), vec.end());})
        .def("__mul__",[](const QubitBasisVector& lhs, const std::complex<double> rhs){return multiply(lhs,rhs);}, py::is_operator())
        .def("__rmul__",[](const QubitBasisVector& lhs, const std::complex<double> rhs){return multiply(lhs,rhs);}, py::is_operator())
        .def("__rmul__",[](const QubitBasisVector& lhs, const PauliString& rhs){return apply(lhs,rhs);}, py::is_operator())
        .def("__rmul__",[](const QubitBasisVector& lhs, const PauliStringVector& rhs){return apply(lhs,rhs);}, py::is_operator())
        .def("__add__",[](const QubitBasisVector& lhs, const QubitBasis& rhs){return add(lhs,rhs);}, py::is_operator())
        .def("__radd__",[](const QubitBasisVector& lhs, const QubitBasis& rhs){return add(lhs,rhs);}, py::is_operator())
        .def("__add__",[](const QubitBasisVector& lhs, const QubitBasisVector& rhs){return add(lhs,rhs);}, py::is_operator())
        .def("__sub__",[](const QubitBasisVector& lhs, const QubitBasis& rhs){return subtract(lhs,rhs);}, py::is_operator())
        .def("__rsub__",[](const QubitBasisVector& lhs, const QubitBasis& rhs){return subtract(rhs,lhs);}, py::is_operator())
        .def("__sub__",[](const QubitBasisVector& lhs, const QubitBasisVector& rhs){return subtract(lhs,rhs);}, py::is_operator())
        .def("get_vec", &QubitBasisVector::get_vec);

    py::class_<PauliString>(m, "PauliString")
        .def(py::init<size_t,size_t, std::complex<double>>(), "X"_a = 0,"Z"_a = 0, "coeff"_a = 1.0, "Make a Pauli string")
        .def("str", &PauliString::str)
        .def("__str__", &PauliString::str)
        .def("__repr__", &PauliString::str)
        .def("__eq__", [](const PauliString& lhs, const PauliString& rhs){return rhs == lhs;})
        .def("__mul__",[](const PauliString& lhs, const PauliString& rhs){return multiply(rhs,lhs);}, py::is_operator())
        .def("__mul__",[](const PauliString& lhs, const std::complex<double> rhs){return multiply(lhs,rhs);}, py::is_operator())
        .def("__rmul__",[](const PauliString& lhs, const std::complex<double> rhs){return multiply(lhs,rhs);}, py::is_operator())
        .def("__add__",[](const PauliString& lhs, const PauliString& rhs){return add(lhs,rhs);}, py::is_operator())
        .def("__sub__",[](const PauliString& lhs, const PauliString& rhs){return subtract(lhs,rhs);}, py::is_operator());

   py::class_<PauliStringVector>(m, "PauliStringVector")
        .def(py::init<std::vector<PauliString>>(), "Make a vector of Pauli string")
        .def("str", &PauliStringVector::str)
        .def("__str__", [](const PauliStringVector& vec){return join(vec.str(), "\n");})
        .def("__repr__", [](const PauliStringVector& vec){return join(vec.str(), "\n");})
        .def("__getitem__", [](const PauliStringVector& vec, unsigned int i){return vec[i];})
        .def("__len__", [](const PauliStringVector& vec){return vec.get_vec().size();})
        .def("__iter__", [](const PauliStringVector& PauliVector){
                std::vector<PauliString> vec(PauliVector.get_vec());
                return py::make_iterator(vec.begin(), vec.end());})
        .def("__mul__",[](const PauliStringVector& lhs, const std::complex<double> rhs){return multiply(lhs,rhs);}, py::is_operator())
        .def("__rmul__",[](const PauliStringVector& lhs, const std::complex<double> rhs){return multiply(lhs,rhs);}, py::is_operator())
        .def("__mul__",[](const PauliStringVector& lhs, const PauliString& rhs){return multiply(lhs,rhs);}, py::is_operator())
        .def("__rmul__",[](const PauliStringVector& lhs, const PauliString& rhs){return multiply(rhs,lhs);}, py::is_operator())
        .def("__mul__",[](const PauliStringVector& lhs, const PauliStringVector& rhs){return multiply(rhs,lhs);}, py::is_operator())
        .def("__add__",[](const PauliStringVector& lhs, const PauliString& rhs){return add(lhs,rhs);}, py::is_operator())
        .def("__radd__",[](const PauliStringVector& lhs, const PauliString& rhs){return add(lhs,rhs);}, py::is_operator())
        .def("__add__",[](const PauliStringVector& lhs, const PauliStringVector& rhs){return add(lhs,rhs);}, py::is_operator())
        .def("__sub__",[](const PauliStringVector& lhs, const PauliString& rhs){return subtract(lhs,rhs);}, py::is_operator())
        .def("__rsub__",[](const PauliStringVector& lhs, const PauliString& rhs){return subtract(rhs,lhs);}, py::is_operator())
        .def("__sub__",[](const PauliStringVector& lhs, const PauliStringVector& rhs){return subtract(lhs,rhs);}, py::is_operator())
        .def("get_vec", &PauliStringVector::get_vec);

    py::class_<Computer>(m, "Computer")
        .def(py::init<size_t>(), "nqubits"_a, "Make a quantum computer with 'nqubits' qubits")
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
        .def("set_state", &Computer::set_state)
        .def("zero_state", &Computer::zero_state)
        .def("get_timings", &Computer::get_timings)
        .def("clear_timings", &Computer::clear_timings)
        .def("str", &Computer::str)
        .def("__str__", [](const Computer& qc) {
            std::string r("Computer(\n");
            for (const std::string& s : qc.str()) {
                r += "  " + s + "\n";
            }
            r += " )";
            return r;
        });

    py::class_<Gate>(m, "Gate")
        .def("target", &Gate::target)
        .def("control", &Gate::control)
        .def("gate_id", &Gate::gate_id)
        .def("sparse_matrix", &Gate::sparse_matrix)
        .def("adjoint", &Gate::adjoint)
        .def("str", &Gate::str)
        .def("__str__", &Gate::str)
        .def("__repr__", &Gate::repr);

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

    m.def("control_gate", &make_control_gate, "control"_a, "Gate"_a);
}
