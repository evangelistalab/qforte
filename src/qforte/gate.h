#ifndef _gate_h_
#define _gate_h_

#include <array>
#include <vector>
#include <map>
#include <set>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "qforte-def.h"

class QubitBasis;
class SparseMatrix;
class SparseVector;

/// useful for type safety and efficient comparisons
enum class GateType {
    X,
    Y,
    Z,
    H,
    R,
    Rx,
    Ry,
    Rz,
    V,
    adjV,
    S,
    T,
    I,
    A,
    cX,
    acX,
    cY,
    cZ,
    cR,
    cV,
    adjcV,
    cRz,
    SWAP,
    Undefined
};

/// alias for a 4 x 4 complex matrix stored as an array of arrays
using complex_4_4_mat = std::array<std::array<std::complex<double>, 4>, 4>;

/// @brief A class representing a quantum gate
/// @details The gate is represented by a 4 x 4 complex matrix, which can represent at most a
/// 2-qubit gate
class Gate {
  public:
    /// @brief Constructor for 1- and 2-qubit gates.
    /// @details It is not recommended to use this constructor directly, but rather use the
    /// make_gate function.
    /// @param label the label for this operator (e.g, "X", "cZ")
    /// @param target the target qubit
    /// @param control the control qubit
    /// @param gate the 4 x 4 matrix representation of the gate
    /// @param parameter the parameter associated with this gate (default: none)
    Gate(const std::string& label, size_t target, size_t control, std::complex<double> gate[4][4],
         std::optional<std::pair<double, bool>> parameter = std::nullopt);

    /// default copy constructor
    Gate(const Gate& gate) = default;

    /// Return the target qubit
    size_t target() const;

    /// Return the control qubit
    size_t control() const;

    /// Returns the 4X4 matrix representation of the gate
    const complex_4_4_mat& matrix() const;

    /// Return true if this gate depends on a parameter
    bool has_parameter() const;

    /// Return the parameter associated with this gate
    std::optional<double> parameter() const;

    /// Return if the parameter should be multiplied by -1 when taking the adjoint
    bool minus_parameter_on_adjoint() const;

    /// Returns the lifted sparse matrix representaion of the gate
    const SparseMatrix sparse_matrix(size_t nqubit) const;

    /// Return a string representation of the gate
    std::string str() const;

    /// Return a string representation of the gate for debugging
    std::string repr() const;

    /// Return the string specifying what type of gate [X, Y , CNOT, ...]
    std::string gate_id() const;

    /// The number of qubits this gate acts on
    size_t nqubits() const;

    /// Return the canonical order of the 2-qubit basis
    static const std::vector<std::pair<size_t, size_t>>& two_qubits_basis();

    // Return the adjoint of this gate
    Gate adjoint() const;

    static const std::vector<size_t>& get_index1() { return index1; }
    static const std::vector<size_t>& get_index2() { return index2; }

    /// Get the gate type
    GateType gate_type() const;

  private:
    /// the label of this gate
    std::string label_;

    /// the gate type
    GateType type_;

    /// the target qubit
    size_t target_;

    /// the control qubit. For single qubit operators control_ == target_;
    size_t control_;

    /// the parameter associated with this gate and a factor indicating if the parameter should be
    /// multiplied by -1 when taking the adjoint
    std::optional<std::pair<double, bool>> parameter_;

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

    /// Determine the GateType based on the label string
    static GateType mapLabelToType(const std::string& label_);
};

/// Gate equality operator
/// @details Two gates are equal if they have the same target, control, and matrix
/// Here we do not check for the label or parameter
bool operator==(const Gate& lhs, const Gate& rhs);
/// Gate inequality operator
bool operator!=(const Gate& lhs, const Gate& rhs);
/// Gate inequality operator
bool operator<(const Gate& lhs, const Gate& rhs);

/// Utility function to create a gate from a string
Gate make_gate(std::string type, size_t target, size_t control, double parameter = 0.0);

/// Utility function to create a controlled version of a gate
Gate make_control_gate(size_t control, Gate& U);

/// Evaluate interaction of two gates
/// bool: whether the gates commute
/// int: which simplification scheme can be applied
std::pair<bool, int> evaluate_gate_interaction(const Gate& gate1, const Gate& gate2);

extern const std::unordered_set<GateType> phase_1qubit_gates;
extern const std::unordered_map<GateType, GateType> controlled_2qubit_to_1qubit_gate;

#endif // _gate_h_
