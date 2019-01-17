#ifndef _quantum_computer_h_
#define _quantum_computer_h_

#include <string>
#include <vector>

#include "qforte-def.h" // double_c

class QuantumGate;

/**
 * @brief The QuantumBasis class
 *
 * This class represents an element of the Hilbert space basis:
 *   |q_1 q_2 ... q_n> with q_i = {0, 1}
 *
 *   for example:
 *
 *   |1010>, |0000>, |1110>
 */
class QuantumBasis {
  public:
    /// the type used to represent a quantum state (a 64 bit unsigned long)
    using basis_t = uint64_t;

    /// the maximum number of qubits
    static constexpr size_t max_qubits_ = 8 * sizeof(basis_t);

    /// constructor
    QuantumBasis(size_t n = static_cast<basis_t>(0)) { state_ = n; }

    /// a mask for bit in position pos
    static constexpr basis_t maskbit(size_t pos) { return (static_cast<basis_t>(1)) << pos; }

    /// get the value of bit pos
    bool get_bit(size_t pos) const { return state_ & maskbit(pos); }

    /// set bit in position 'pos' to the boolean val
    basis_t& set_bit(size_t pos, bool val) {
        if (val)
            state_ |= maskbit(pos);
        else
            state_ &= ~maskbit(pos);
        return state_;
    }

    void set(basis_t state);
    void zero() { state_ = static_cast<basis_t>(0); }

    QuantumBasis& insert(size_t pos);

    size_t add() const { return state_; }

    std::string str(size_t nqubit) const;

  private:
    /// the state
    basis_t state_;
};

class QuantumCircuit {
  public:
    /// default constructor: creates an empty circuit
    QuantumCircuit() {}

    /// add a gate
    void add_gate(const QuantumGate& gate) { gates_.push_back(gate); }

    /// return a vector of gates
    const std::vector<QuantumGate>& gates() const { return gates_; }

    /// return a vector of string representing this circuit
    std::vector<std::string> str() const;

  private:
    /// the list of gates
    std::vector<QuantumGate> gates_;
};

class QuantumComputer {
  public:
    /// default constructor: create a quantum computer with nqubit qubits
    QuantumComputer(int nqubit);

    /// apply a quantum circuit to the current state
    void apply_circuit(const QuantumCircuit& qc);

    /// apply a gate to the quantum computer
    void apply_gate(const QuantumGate& qg);

    /// measure the expectation value of many 1qubit gates
    std::complex<double> measure_circut(QuantumCircuit& qc);

    /// measure the expectation value of a single 1qubit gate
    std::complex<double> measure_gate(QuantumGate& qg);

    /// return a vector of strings representing the state of the computer
    std::vector<std::string> str() const;

    /// return the coefficient of a basis state
    std::complex<double> coeff(const QuantumBasis& basis);

    void set_state(std::vector<std::pair<QuantumBasis, double_c>> state);

  private:
    /// the number of qubits
    size_t nqubit_;
    /// the number of basis states
    size_t nbasis_;
    /// the tensor product basis
    std::vector<QuantumBasis> basis_;
    /// the coefficients of the tensor product basis
    std::vector<std::complex<double>> coeff_;
    /// the coefficients of the tensor product basis
    std::vector<std::complex<double>> new_coeff_;

    /// the threshold for priting a determinant
    double print_threshold_ = 0.0;

    double compute_threshold_ = 1.0e-16;

    void apply_1qubit_gate(const QuantumGate& qg);
    void apply_1qubit_gate_insertion(const QuantumGate& qg);

    void apply_2qubit_gate(const QuantumGate& qg);
};

#endif // _quantum_computer_h_
