#ifndef _circuit_h_
#define _circuit_h_

#include <vector>
#include <map>
#include <complex>

#include <iostream>

class Gate;
class SparseMatrix;

class Circuit {
    /* A Circuit is a product of quantum gates. An individual gate acts on
     * at most one or two qubits and must be of the forms we can efficiently prepare,
     * e.g., CNOT, Hadamard, Rotation.
     * */
  public:
    /// default constructor: creates an empty circuit
    Circuit() {}

    /// add a gate
    void add_gate(const Gate& gate) { gates_.push_back(gate); }

    /// add a circuit
    void add_circuit(const Circuit& circ);

    /// return a vector of gates
    const std::vector<Gate>& gates() const { return gates_; }

    /// return the number of gates
    size_t size() const { return gates_.size(); }

    /// return the adjoint (conjugate transpose) of this Circuit
    Circuit adjoint();

    /// reset the circuit with a new set of parameters
    void set_parameters(const std::vector<double>& params);

    /// For a circuit of Pauli gates, orders gates from those with smallest-index
    /// target to largest-index target AND combines gates with same target.
    /// Returns the prefactor resulting from the gate combinations (+/- 1.0 or +/- 1.0j).
    std::complex<double> canonicalize_pauli_circuit();

    /// get the number of CNOT gates in the circuit
    int get_num_cnots() const;

    /// Returns the lifted sparse matrix representaion of the circuit,
    /// the matrix should always be unitary.
    const SparseMatrix sparse_matrix(size_t nqubit) const;

    /// Return a vector of string representing this circuit.
    std::string str() const;

    /// Return the number of qubits pertaining to this circuit. Note this is
    /// not the numebr of unique qubits but the minimum number of qubits needed
    /// to execute the circuit. For example the circut [X_0 Y_4 X_8] would requre
    /// nine qubits.    
    size_t num_qubits() const;

  private:
    /// the list of gates
    std::vector<Gate> gates_;
};

// A equivalence comparison for Circuit class
bool operator==(const Circuit& qc1, const Circuit& qc2);



#endif // _circuit_h_
