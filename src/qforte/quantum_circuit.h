#ifndef _quantum_circuit_h_
#define _quantum_circuit_h_

#include <vector>
#include <map>
#include <complex>

#include <iostream>

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

    /// reorders the circuit by increading qubit and retruns the resluting factor
    /// after contracting all pauli gates
    /// (either +/-1.0 or +/-1.0j)
    std::complex<double> canonical_order();

    /// return a vector of string representing this circuit
    std::string str() const;

  private:
    /// the list of gates
    std::vector<QuantumGate> gates_;
};

// A eqivalence comparitor for QuantumCircuit class
bool operator==(const QuantumCircuit& qc1, const QuantumCircuit& qc2);



#endif // _quantum_circuit_h_
