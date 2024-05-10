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
    /// default constructor
    Circuit() = default;
    /// default copy constructor
    Circuit(const Circuit& other) = default;

    /// append a gate at the end of this circuit
    void add_gate(const Gate& gate) { gates_.push_back(gate); }

    /// insert a gate at a given position in the circuit
    void insert_gate(size_t pos, const Gate& gate);

    /// remove a gate at a given position in the circuit
    void remove_gate(size_t pos);

    /// swap two gates in the circuit
    void swap_gates(size_t pos1, size_t pos2);

    /// append a circuit at the end of this circuit
    void add_circuit(const Circuit& circ);

    /// insert a circuit at a given position in the circuit
    void insert_circuit(size_t pos, const Circuit& circ);

    /// remove gates in interval [pos1, pos2)
    void remove_gates(size_t pos1, size_t pos2);

    /// replace a gate at a given position in the circuit
    void replace_gate(size_t pos, const Gate& gate);

    /// return a vector of gates
    const std::vector<Gate>& gates() const { return gates_; }

    /// return a vector of gates
    const Gate& gate(size_t n) const { return gates_[n]; }

    /// return the number of gates
    size_t size() const { return gates_.size(); }

    /// return the adjoint (conjugate transpose) of this Circuit
    Circuit adjoint();

    /// reset the circuit with a new set of parameters
    void set_parameters(const std::vector<double>& params);

    /// return the parameters of this circuit
    std::vector<double> get_parameters() const;

    /// update the parameters of a single gate
    void set_parameter(size_t pos, double param);

    /// For a circuit of Pauli gates, orders gates from those with smallest-index
    /// target to largest-index target AND combines gates with same target.
    /// Returns the prefactor resulting from the gate combinations (+/- 1.0 or +/- 1.0j).
    std::complex<double> canonicalize_pauli_circuit();

    /// get the number of CNOT gates in the circuit
    int get_num_cnots() const;

    /// Returns the lifted sparse matrix representaion of the circuit,
    /// the matrix should always be unitary.
    const SparseMatrix sparse_matrix(size_t nqubit) const;

    /// Return a vector of string representing this circuit
    std::string str() const;

    /// Return the number of qubits pertaining to this circuit. Note this is
    /// not the numebr of unique qubits but the minimum number of qubits needed
    /// to execute the circuit. For example the circut [X_0 Y_4 X_8] would requre
    /// nine qubits.
    size_t num_qubits() const;

    /// Return true if this circuit is composed of only Pauli gates
    bool is_pauli() const;

    /// Simplify the circuit
    void simplify();

  private:
    /// the list of gates
    std::vector<Gate> gates_;

    /// helper function to simplify phase gates
    double get_phase_gate_parameter(const Gate& gate);
};

bool operator==(const Circuit& qc1, const Circuit& qc2);
bool operator<(const Circuit& qc1, const Circuit& qc2);

#endif // _circuit_h_
