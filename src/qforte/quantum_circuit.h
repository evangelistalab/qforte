#ifndef _quantum_circuit_h_
#define _quantum_circuit_h_

#include <vector>

class QuantumGate;

class QuantumCircuit {
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

    /// return a vector of string representing this circuit
    std::vector<std::string> str() const;

  private:
    /// the list of gates
    std::vector<QuantumGate> gates_;

    /// reversed list of gates
    std::vector<QuantumGate> rev_copy_;
};

#endif // _quantum_circuit_h_
