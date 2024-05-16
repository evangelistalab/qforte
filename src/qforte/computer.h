#ifndef _computer_h_
#define _computer_h_

#include <string>
#include <vector>

#include "qforte-def.h" // double_c

template <class T> std::complex<T> complex_prod(std::complex<T> a, std::complex<T> b) {
    return std::conj<T>(a) * b;
}

template <class T> std::complex<T> add_c(std::complex<T> a, std::complex<T> b) { return a + b; }

class Gate;
class QubitBasis;
class Circuit;
class QubitOperator;
class QubitOpPool;
class SparseMatrix;

class Computer {
  public:
    /// default constructor: create a quantum computer with nqubit qubits
    Computer(int nqubit, double print_threshold = 1.0e-6);

    /// default copy constructor
    Computer(const Computer& other) = default;

    /// applies a matrix representation of a Gate, Gircuit, or QubitOoperator
    /// to the quantum state.
    void apply_matrix(const std::vector<std::vector<std::complex<double>>>& Opmat);

    /// applies a sparse matrix representation of a Gate, Gircuit, or QubitOoperator
    /// to the quantum state.
    void apply_sparse_matrix(const SparseMatrix& Spmat);

    /// apply a quantum operator to the current state with optimized algorithm
    /// (this operation is generally not a physical quantum computing operation).
    void apply_operator(const QubitOperator& qo);

    /// apply a quantum circuit to the current state with standard algorithm
    void apply_circuit_safe(const Circuit& qc);

    /// apply a quantum circuit to the current state with optimized algorithm
    void apply_circuit(const Circuit& qc);

    /// apply a gate to the quantum computer with standard algorithm
    void apply_gate_safe(const Gate& qg);

    /// apply a gate to the quantum computer with optimized algorithm
    void apply_gate(const Gate& qg);

    /// apply a constant to the quantum computer (WARNING, this operation
    /// is not physical as it does not represent a unitary opperation). Only
    /// Exists for 'fast' version of the algorithm for efficiency reasons
    void apply_constant(const std::complex<double> a);

    /// measure the state of the quantum computer with respect to qc
    std::vector<double> measure_circuit(const Circuit& qc, size_t n_measurements);

    /// measure the readout, i.e. the value of all qubits with indices from na to nb
    std::vector<std::vector<int>> measure_z_readouts_fast(size_t na, size_t nb,
                                                          size_t n_measurements);

    /// measure the readout, i.e. the value of all target qubits, for the state of the
    /// quanum computer with respect to qc
    std::vector<std::vector<int>> measure_readouts(const Circuit& qc, size_t n_measurements);

    /// perfectly measure the state of the quanum computer in basis of circuit
    double perfect_measure_circuit(const Circuit& qc);

    /// Measure expectation value of all operators in an operator pool
    std::vector<std::complex<double>> direct_oppl_exp_val(const QubitOpPool& qopl);

    /// measure expectation value for specific operators in an operator pool
    std::vector<std::complex<double>> direct_idxd_oppl_exp_val(const QubitOpPool& qopl,
                                                               const std::vector<int>& idxs);

    /// measure expectaion value of all operators in an operator pool, where the
    /// operator coefficents have been multipild by mults
    std::vector<std::complex<double>>
    direct_oppl_exp_val_w_mults(const QubitOpPool& qopl,
                                const std::vector<std::complex<double>>& mults);

    /// get the expectation value of the sum of many circuits directly
    /// (ie without simulated measurement)
    std::complex<double> direct_op_exp_val(const QubitOperator& qo);

    /// get the expectation value of many 1qubit gates directly
    /// (ie without simulated measurement)
    std::complex<double> direct_circ_exp_val(const Circuit& qc);

    /// get the expectation value of many pauli gates directly
    /// (ie without simulated measurement)
    std::complex<double> direct_pauli_circ_exp_val(const Circuit& qc);

    /// get the idx I with respect to pauli circuit permutations from qc
    std::pair<size_t, std::complex<double>> get_pauli_permuted_idx(size_t I,
                                                                   const std::vector<int>& x_idxs,
                                                                   const std::vector<int>& y_idxs,
                                                                   const std::vector<int>& z_idxs);

    /// get the expectation value of a single 1qubit gate directly
    /// (without simulated measurement)
    std::complex<double> direct_gate_exp_val(const Gate& qg);

    /// return a string representing the state of the computer
    std::string str() const;

    /// return a vector of the coefficients
    const std::vector<std::complex<double>>& get_coeff_vec() const { return coeff_; };

    /// return the vector of basis states
    const std::vector<QubitBasis>& get_basis_vec() const { return basis_; };

    /// return the coefficient of a basis state
    std::complex<double> coeff(const QubitBasis& basis);

    /// return the number of qubits
    size_t get_nqubit() const { return nqubit_; }

    /// return the number of basis states
    size_t get_nbasis() const { return nbasis_; }
    /// return the number of one-qubit operations
    size_t none_ops() const { return none_ops_; }

    /// return the number of two-qubit operations
    size_t ntwo_ops() const { return ntwo_ops_; }

    // set the coefficient vector directly from another coefficient vector
    void set_coeff_vec(const std::vector<double_c>& c_vec) { coeff_ = c_vec; }

    /// set the quantum computer to the state
    /// basis_1 * c_1 + basis_2 * c_2 + ...
    /// where this information is passed as a vectors of pairs
    ///  [(basis_1, c_1), (basis_2, c_2), ...]
    void set_state(std::vector<std::pair<QubitBasis, double_c>> state);

    /// reset the quantum computer to the null state (all coefficients are 0)
    void null_state();

    /// reset the quantum computer to the state |0>
    void reset();

    /// get timings
    std::vector<std::pair<std::string, double>> get_timings() { return timings_; }

    /// clear the timings
    void clear_timings() { timings_.clear(); }

  private:
    /// the number of qubits
    size_t nqubit_;
    /// the number of basis states (2 ^ nqubit_)
    size_t nbasis_;
    /// the tensor product basis
    std::vector<QubitBasis> basis_;
    /// The coefficients of the starting state in the tensor product basis
    std::vector<std::complex<double>> coeff_;
    /// the coefficients of the ending state in the tensor product basis
    std::vector<std::complex<double>> new_coeff_;
    /// timings and descriptions accessable in python
    std::vector<std::pair<std::string, double>> timings_;
    /// the number of one-qubit operations (used for tracking operations)
    size_t none_ops_ = 0;
    /// the number of two-qubit operations (used for tracking operations)
    size_t ntwo_ops_ = 0;
    /// the threshold for priting a determinant
    double print_threshold_;
    /// the threshold for doing operations with elements of gate matricies
    double compute_threshold_ = 1.0e-16;

    /// apply a 1qubit gate to the quantum computer with standard algorithm
    void apply_1qubit_gate_safe(const Gate& qg);

    /// apply a 1qubit gate to the quantum computer with optimized algorithm
    void apply_1qubit_gate(const Gate& qg);

    /// apply a 2qubit gate to the quantum computer with standard algorithm
    void apply_2qubit_gate_safe(const Gate& qg);

    /// apply a 2qubit gate to the quantum computer with optimized algorithm
    void apply_2qubit_gate(const Gate& qg);
};

/// test the equality of two quantum computers
bool operator==(const Computer& qc1, const Computer& qc2);

/// return the dot product of two quantum computers
std::complex<double> dot(const Computer& qc1, const Computer& qc2);

#endif // _computer_h_
