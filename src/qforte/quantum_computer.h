#ifndef _quantum_computer_h_
#define _quantum_computer_h_

#include <string>
#include <vector>

#include "qforte-def.h" // double_c

template <class T> std::complex<T> complex_prod(std::complex<T> a, std::complex<T> b) {
    return std::conj<T>(a) * b;
}

template <class T> std::complex<T> add_c(std::complex<T> a, std::complex<T> b) { return a + b; }

class QuantumGate;
class QuantumBasis;
class QuantumCircuit;
class QuantumOperator;

class QuantumComputer {
  public:
    /// default constructor: create a quantum computer with nqubit qubits
    QuantumComputer(int nqubit);

    /// apply a quantum circuit to the current state
    void apply_circuit(const QuantumCircuit& qc);

    /// apply a gate to the quantum computer
    void apply_gate(const QuantumGate& qg);

    /// apply a gate to the quantum computer
    void apply_gate_fast(const QuantumGate& qg);

    /// measure the state of the quanum computer in basis of circuit
    std::vector<double> measure_circuit(const QuantumCircuit& qc, size_t n_measurements);

    /// get the expectation value of the sum of many circuits directly
    /// (ie without simulated measurement)
    std::complex<double> direct_op_exp_val(const QuantumOperator& qo);

    /// get the expectation value of many 1qubit gates directly
    /// (ie without simulated measurement)
    std::complex<double> direct_circ_exp_val(const QuantumCircuit& qc);

    /// get the expectation value of a single 1qubit gate directly
    /// (ie without simulated measurement)
    std::complex<double> direct_gate_exp_val(const QuantumGate& qg);

    /// return a vector of strings representing the state of the computer
    std::vector<std::string> str() const;

    /// return a vector of the coeficients
    std::vector<std::complex<double>> get_coeff_vec() const { return coeff_; };

    /// return the coefficient of a basis state
    std::complex<double> coeff(const QuantumBasis& basis);

    /// return the number of qubits
    size_t get_nqubit() const { return nqubit_; }

    /// return the number of basis states
    size_t get_nbasis() const { return nbasis_; }

    /// return the number of one-qubit operations
    size_t none_ops() const { return none_ops_; }

    /// return the number of two-qubit operations
    size_t ntwo_ops() const { return ntwo_ops_; }

    /// set the quantum computer to the state
    /// basis_1 * c_1 + basis_2 * c_2 + ...
    /// where this information is passed as a vectors of pairs
    ///  [(basis_1, c_1), (basis_2, c_2), ...]
    void set_state(std::vector<std::pair<QuantumBasis, double_c>> state);

    void zero_state();

  private:
    /// the number of qubits
    size_t nqubit_;
    /// the number of basis states (2 ^ nqubit_)
    size_t nbasis_;
    /// the tensor product basis
    std::vector<QuantumBasis> basis_;
    /// the coefficients of the tensor product basis
    std::vector<std::complex<double>> coeff_;
    /// the coefficients of the tensor product basis
    std::vector<std::complex<double>> new_coeff_;
    /// the number of one-qubit operations
    size_t none_ops_ = 0;
    /// the number of two-qubit operations
    size_t ntwo_ops_ = 0;

    /// the threshold for priting a determinant
    double print_threshold_ = 0.0;

    double compute_threshold_ = 1.0e-16;

    void apply_1qubit_gate(const QuantumGate& qg);

    void apply_1qubit_gate_fast(const QuantumGate& qg);

    void apply_2qubit_gate(const QuantumGate& qg);
};

#endif // _quantum_computer_h_
