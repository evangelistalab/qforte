#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "fmt/format.h"

#include "qubit_basis.h"
#include "circuit.h"
#include "gate.h"
#include "computer.h"
#include "fci_computer.h"
#include "fci_graph.h"
#include "qubit_operator.h"
#include "sq_operator.h"
#include "sq_op_pool.h"
#include "qubit_op_pool.h"
#include "sparse_tensor.h"
#include "timer.h"
#include "tensor.h"
#include "tensor_operator.h"
#include "blas_math.h"

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
        .def("mult_coeffs", &SQOperator::mult_coeffs)
        .def("terms", &SQOperator::terms)
        .def("get_largest_alfa_beta_indices", &SQOperator::get_largest_alfa_beta_indices) // TODO(Tyler) Need Test
        .def("many_body_order", &SQOperator::many_body_order) // TODO(Tyler) Need Test
        .def("ranks_present", &SQOperator::ranks_present) // TODO(Tyler) Need Test
        .def("canonical_order", &SQOperator::canonical_order)
        .def("simplify", &SQOperator::simplify)
        .def("jw_transform", &SQOperator::jw_transform, py::arg("qubit_excitation") = false)
        .def("split_by_rank", &SQOperator::split_by_rank)
        .def("str", &SQOperator::str)
        .def("__str__", &SQOperator::str)
        .def("__repr__", &SQOperator::str);

    py::class_<TensorOperator>(m, "TensorOperator")
        .def(py::init<size_t, size_t, bool, bool>(), 
            "max_nbody"_a, 
            "dim"_a, 
            "is_spatial"_a=false, 
            "is_restricted"_a=false, 
            "Make a TensorOperator")
        .def("add_sqop_of_rank", &TensorOperator::add_sqop_of_rank)    
        .def("tensors", &TensorOperator::tensors)
        .def("fill_tensor_from_np_by_rank", &TensorOperator::fill_tensor_from_np_by_rank)
        .def("str", &TensorOperator::str, 
            py::arg("print_data") = true, 
            py::arg("print_complex") = false, 
            py::arg("maxcols") = 5,
            py::arg("data_format") = "%12.7f",
            py::arg("header_format") = "%12zu")
        .def("__str__", &TensorOperator::str,
            py::arg("print_data") = true, 
            py::arg("print_complex") = false, 
            py::arg("maxcols") = 5,
            py::arg("data_format") = "%12.7f",
            py::arg("header_format") = "%12zu")
        .def("__repr__", &TensorOperator::str,
            py::arg("print_data") = true, 
            py::arg("print_complex") = false, 
            py::arg("maxcols") = 5,
            py::arg("data_format") = "%12.7f",
            py::arg("header_format") = "%12zu");

    py::class_<SQOpPool>(m, "SQOpPool")
        .def(py::init<>())
        .def("add", &SQOpPool::add_term)
        .def("add_hermitian_pairs", &SQOpPool::add_hermitian_pairs)
        .def("add_term", &SQOpPool::add_term)
        .def("set_coeffs", &SQOpPool::set_coeffs)
        .def("set_coeffs_to_scaler", &SQOpPool::set_coeffs_to_scaler)
        .def("terms", &SQOpPool::terms)
        .def("set_orb_spaces", &SQOpPool::set_orb_spaces)
        .def("get_qubit_op_pool", &SQOpPool::get_qubit_op_pool)
        .def("get_qubit_operator", &SQOpPool::get_qubit_operator, py::arg("order_type"),
             py::arg("combine_like_terms") = true, py::arg("qubit_excitations") = false)
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
        .def("__iter__", [](const QubitOperator &op) { return py::make_iterator(op.terms()); },
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
        .def("__iter__", [](const QubitOpPool &pool) { return py::make_iterator(pool.terms()); },
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
        .def("add", &QubitBasis::add)
        .def("get_bit", &QubitBasis::get_bit);

    py::class_<Computer>(m, "Computer")
        .def(py::init<size_t,double>(), "nqubits"_a, "print_threshold"_a = 1.0e-6, "Make a quantum computer with 'nqubits' qubits")
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
        .def("apply_sq_operator", &Computer::apply_sq_operator)
        .def("z_chain", &Computer::z_chain)
        .def("apply_2x2", &Computer::apply_2x2)
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
        .def("__repr__", &Computer::str);

    py::class_<FCIComputer>(m, "FCIComputer")
        .def(py::init<int, int, int>(), "nel"_a, "sz"_a, "norb"_a, "Make a FCIComputer with nel, sz, and norb")
        .def("hartree_fock", &FCIComputer::hartree_fock)
        .def("apply_tensor_spin_1bdy", &FCIComputer::apply_tensor_spin_1bdy)
        .def("apply_tensor_spin_12bdy", &FCIComputer::apply_tensor_spin_12bdy)
        .def("apply_tensor_spin_012bdy", &FCIComputer::apply_tensor_spin_012bdy)
        .def("apply_tensor_spat_12bdy", &FCIComputer::apply_tensor_spat_12bdy)
        .def("apply_tensor_spat_012bdy", &FCIComputer::apply_tensor_spat_012bdy)
        .def("apply_individual_sqop_term", &FCIComputer::apply_individual_sqop_term)
        .def("apply_sqop", &FCIComputer::apply_sqop)
        .def("apply_sqop_pool", &FCIComputer::apply_sqop_pool)
        .def("get_exp_val", &FCIComputer::get_exp_val)
        .def("get_exp_val_tensor", &FCIComputer::get_exp_val_tensor)
        .def("evolve_op_taylor", &FCIComputer::evolve_op_taylor)
        .def("apply_sqop_evolution", &FCIComputer::apply_sqop_evolution, 
            py::arg("time"),
            py::arg("sqop"),
            py::arg("antiherm") = false,
            py::arg("adjoint") = false
            )
        .def("evolve_pool_trotter_basic", &FCIComputer::evolve_pool_trotter_basic, 
            py::arg("sqop"),
            py::arg("antiherm") = false,
            py::arg("adjoint") = false
            )
        .def("evolve_pool_trotter", &FCIComputer::evolve_pool_trotter, 
            py::arg("sqop"),
            py::arg("evolution_time"),
            py::arg("trotter_steps"),
            py::arg("trotter_order"),
            py::arg("antiherm") = false,
            py::arg("adjoint") = false
            )
        .def("set_state", &FCIComputer::set_state)
        .def("get_state", &FCIComputer::get_state)
        .def("get_state_deep", &FCIComputer::get_state_deep)
        .def("get_hf_dot", &FCIComputer::get_hf_dot)
        .def("str", &FCIComputer::str, 
            py::arg("print_data") = true, 
            py::arg("print_complex") = false)
        .def("__str__", &FCIComputer::str, 
            py::arg("print_data") = true, 
            py::arg("print_complex") = false)
        .def("__repr__", &FCIComputer::str, 
            py::arg("print_data") = true, 
            py::arg("print_complex") = false);

    py::class_<FCIGraph>(m, "FCIGraph")
        .def(py::init<int, int, int>(), "nalfa"_a, "nbeta"_a, "norb"_a, "Make a FCIGraph")
        .def("make_mapping_each", &FCIGraph::make_mapping_each)
        .def("get_nalfa", &FCIGraph::get_nalfa)
        .def("get_nbeta", &FCIGraph::get_nbeta)
        .def("get_lena", &FCIGraph::get_lena)
        .def("get_lenb", &FCIGraph::get_lenb)
        .def("get_astr", &FCIGraph::get_astr)
        .def("get_bstr", &FCIGraph::get_bstr)
        .def("get_aind", &FCIGraph::get_aind)
        .def("get_bind", &FCIGraph::get_bind)
        .def("get_alfa_map", &FCIGraph::get_alfa_map)
        .def("get_beta_map", &FCIGraph::get_beta_map)
        .def("get_dexca", &FCIGraph::get_dexca)
        .def("get_dexcb", &FCIGraph::get_dexcb)
        .def("get_dexca_vec", &FCIGraph::get_dexca_vec)
        .def("get_dexcb_vec", &FCIGraph::get_dexcb_vec);


    py::class_<Tensor>(m, "Tensor")
        .def(py::init<std::vector<size_t>, std::string>(), "shape"_a, "name"_a, "Make a Tensor with a particualr shape")
        .def(py::init<>())
        .def("name", &Tensor::name)
        .def("ndim", &Tensor::ndim)
        .def("size", &Tensor::size)
        .def("shape", &Tensor::shape)
        .def("strides", &Tensor::strides)
        .def("set", &Tensor::set)
        .def("copy_in", &Tensor::copy_in)
        .def("add_to_element", &Tensor::add_to_element)
        .def("get", &Tensor::get)
        .def("fill_from_np", &Tensor::fill_from_np)
        .def("add", &Tensor::add) // TODO(Tyler) Need Test (use numpy)
        .def("subtract", &Tensor::subtract)
        .def("norm", &Tensor::norm)
        .def("scale", &Tensor::scale) // TODO(Tyler) Need Test (use numpy)
        .def("identity", &Tensor::identity) // TODO(Tyler) Need Test 
        .def("zero", &Tensor::zero) // TODO(Tyler) Need Test 
        .def("zero_with_shape", &Tensor::zero_with_shape) // TODO(Tyler) Need Test 
        .def("vector_dot", &Tensor::vector_dot) // TODO(Tyler) Need Test 
        .def("symmetrize", &Tensor::symmetrize) // TODO(Tyler) Need Test 
        .def("antisymmetrize", &Tensor::antisymmetrize) // TODO(Tyler) Need Test 
        .def("transpose", &Tensor::transpose) // TODO(Tyler) Need Test (use numpy)
        .def("general_transpose", &Tensor::general_transpose) // TODO(Tyler) Need Test (use numpy)
        .def("fill_from_nparray", &Tensor::fill_from_nparray)
        .def("zaxpy", &Tensor::zaxpy, "x"_a, "alpha"_a, "incx"_a = 1, "incy"_a = 1) // TODO(Tyler) Need Test (use numpy)
        .def("zaxpby", &Tensor::zaxpby, "x"_a, "a"_a, "b"_a, "incx"_a = 1, "incy"_a = 1)
        .def("gemm", &Tensor::gemm, "B"_a, 
            "transa"_a = 'N', 
            "transb"_a = 'N', 
            "alpha"_a = 1.0, 
            "beta"_a = 1.0, 
            "mult_B_on_right"_a = false)


        .def_static("chain", &Tensor::chain, "As"_a, "trans"_a, "alpha"_a = 1.0, "beta"_a = 0.0) // TODO(Tyler) Need Test (use numpy)

        .def_static("einsum", 
            &Tensor::einsum, 
            "Ainds"_a, 
            "Binds"_a, 
            "Cinds"_a, 
            "A"_a,
            "B"_a, 
            "C3"_a,
            "alpha"_a = 1.0,
            "beta"_a = 0.0) 

        .def_static("permute", 
            &Tensor::permute, 
            "Ainds"_a, 
            "Cinds"_a, 
            "A"_a,
            "C2"_a,
            "alpha"_a = 1.0,
            "beta"_a = 0.0) 

        .def("slice", &Tensor::slice)
        .def("get_nonzero_tidxs", &Tensor::get_nonzero_tidxs)

        .def("str", &Tensor::str, 
            py::arg("print_data") = true, 
            py::arg("print_complex") = false, 
            py::arg("maxcols") = 5,
            py::arg("data_format") = "%12.7f",
            py::arg("header_format") = "%12zu")
        .def("__str__", &Tensor::str,
            py::arg("print_data") = true, 
            py::arg("print_complex") = false, 
            py::arg("maxcols") = 5,
            py::arg("data_format") = "%12.7f",
            py::arg("header_format") = "%12zu")
        .def("__repr__", &Tensor::str,
            py::arg("print_data") = true, 
            py::arg("print_complex") = false, 
            py::arg("maxcols") = 5,
            py::arg("data_format") = "%12.7f",
            py::arg("header_format") = "%12zu");

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
        .def("get", &local_timer::get)
        .def("record", &local_timer::record)
        .def("__str__", &local_timer::str_table);

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
            auto vec = {"SWAP", "cV", "CNOT", "cX", "aCNOT", "acX","cY", "cZ"};
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
