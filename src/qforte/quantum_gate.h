#ifndef _quantum_gate_h_
#define _quantum_gate_h_

#include <array>
#include <vector>

#include "qforte-def.h" // double_c

/// alias for a 4 x 4 complex matrix
using complex_4_4_mat = std::array<std::array<std::complex<double>, 4>, 4>;

class QuantumGate {
  public:
    QuantumGate(const std::string& label, size_t target, size_t control,
                std::complex<double> gate[4][4]);

    /// Return the target qubit
    size_t target() const;

    /// Return the control qubit
    size_t control() const;

    const complex_4_4_mat& gate() const;

    std::string str() const;

    size_t nqubits() const;

    static const std::vector<std::pair<size_t, size_t>>& two_qubits_basis();

  private:
    /// the label of this gate
    std::string label_;
    /// the target qubit
    size_t target_;
    /// the control qubit. For single qubit operators control_ == target_;
    size_t control_;
    /// the matrix representatin of this gate
    complex_4_4_mat gate_;

    static const std::vector<std::pair<size_t, size_t>> two_qubits_basis_;
    static const std::vector<size_t> index1;
    static const std::vector<size_t> index2;
};

/// Create a quantum gate
QuantumGate make_gate(std::string type, size_t target, size_t control,
                              double parameter = 0.0);

#endif // _quantum_gate_h_
