#ifndef _quantum_circuit_h_
#define _quantum_circuit_h_

#include <vector>
#include <map>
#include <complex>

#include <iostream>

class QuantumGate;

class QuantumCircuit {
    /* A QuantumCircuit is a product of quantum gates. An individual gate acts on 
     * at most one or two qubits and must be of the forms we can efficiently prepare,
     * e.g., CNOT, Hadamard, Rotation.
     * */
  public:
    /// default constructor: creates an empty circuit
    QuantumCircuit() {}

    /// add a gate
    void add_gate(const QuantumGate& gate) { gates_.push_back(gate); }

    /// add a circuit
    void add_circuit(const QuantumCircuit& circ);

    /// return a vector of gates
    const std::vector<QuantumGate>& gates() const { return gates_; }

    /// return the number of gates
    size_t size() const { return gates_.size(); }

    /// return the adjoint (conjugate transpose) of this QuantumCircuit
    QuantumCircuit adjoint();

    /// reset the circuit with a new set of parameters
    void set_parameters(const std::vector<double>& params);

    /// For a circuit of Pauli gates, orders gates from those with smallest-index
    /// target to largest-index target AND combines gates with same target.
    /// Returns the prefactor resulting from the gate combinations (+/- 1.0 or +/- 1.0j).
    std::complex<double> canonicalize_pauli_circuit();

    /// get the number of CNOT gates in the circuit
    int get_num_cnots() const;

    /// return a vector of string representing this circuit
    std::string str() const;

    size_t num_qubits() const;

  private:
    /// the list of gates
    std::vector<QuantumGate> gates_;
};

// A eqivalence comparitor for QuantumCircuit class
bool operator==(const QuantumCircuit& qc1, const QuantumCircuit& qc2);



#endif // _quantum_circuit_h_
