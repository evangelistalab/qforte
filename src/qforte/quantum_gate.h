#ifndef _quantum_gate_h_
#define _quantum_gate_h_

#include <array>
#include <vector>

#include "qforte-def.h" // double_c

/// alias for a 4 x 4 complex matrix stored as an array of arrays
using complex_4_4_mat = std::array<std::array<std::complex<double>, 4>, 4>;

class QuantumGate {
  public:
    /**
     * @brief QuantumGate
     * @param label the label for this operator (e.g, "X", "cZ")
     * @param target the target qubit
     * @param control the control qubit
     * @param gate the 4 x 4 matrix representation of the gate
     */
    QuantumGate(const std::string& label, size_t target, size_t control,
                std::complex<double> gate[4][4]);

    /// Return the target qubit
    size_t target() const;

    /// Return the control qubit
    size_t control() const;

    const complex_4_4_mat& gate() const;

    /// Return a string representation of the gate
    std::string str() const;

    /// Return a string representation of the gate for debugging
    std::string repr() const;

    /// Return the string specifying what type of gate [X, Y , CNOT, ...]
    std::string gate_id() const;

    size_t nqubits() const;

    static const std::vector<std::pair<size_t, size_t>>& two_qubits_basis();

    // Return the adjoint of this gate
    QuantumGate adjoint() const;

  private:
    /// the label of this gate
    std::string label_;
    /// the target qubit
    size_t target_;
    /// the control qubit. For single qubit operators control_ == target_;
    size_t control_;
    /// the matrix representatin of this gate.
    /// 1 qubit operators are represented by the top left 2 x 2 submatrix.
    complex_4_4_mat gate_;

    /// This vector stores the canonical order of the 2-qubit basis, namely:
    /// control   target
    ///     |0> x |0>
    ///     |0> x |1>
    ///     |1> x |0>
    ///     |1> x |1>
    static const std::vector<std::pair<size_t, size_t>> two_qubits_basis_;
    /// This vector contains the indices of a 1-qubit basis [0, 1]
    static const std::vector<size_t> index1;
    /// This vector contains the indices of a 2-qubit basis [0, 1, 2, 3] where
    /// 0 -> |00>
    /// 1 -> |01>
    /// 2 -> |10>
    /// 3 -> |11>
    static const std::vector<size_t> index2;
};

/// Create a quantum gate
// QuantumGate make_gate(std::string type, size_t target, size_t control,
//                               double parameter = 0.0, bool mirror = false);
QuantumGate make_gate(std::string type, size_t target, size_t control, std::complex<double> parameter = 0.0);

QuantumGate make_control_gate(size_t control, QuantumGate& U);

#endif // _quantum_gate_h_
