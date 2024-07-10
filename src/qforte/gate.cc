#include "fmt/format.h"

#include "gate.h"
#include "qubit_basis.h"
#include "sparse_tensor.h"

const std::vector<std::pair<size_t, size_t>> Gate::two_qubits_basis_{
    {0, 0}, {0, 1}, {1, 0}, {1, 1}};
const std::vector<size_t> Gate::index1{0, 1};
const std::vector<size_t> Gate::index2{0, 1, 2, 3};

Gate::Gate(const std::string& label, size_t target, size_t control, std::complex<double> gate[4][4],
           std::optional<std::pair<double, bool>> parameter)
    : label_(label), target_(target), control_(control), parameter_(parameter) {
    for (const auto& i : index2) {
        for (const auto& j : index2) {
            gate_[i][j] = gate[i][j];
        }
    }
    type_ = mapLabelToType(label_);
}

size_t Gate::target() const { return target_; }

size_t Gate::control() const { return control_; }

const complex_4_4_mat& Gate::matrix() const { return gate_; }

const SparseMatrix Gate::sparse_matrix(size_t nqubit) const {
    if (target_ >= nqubit) {
        throw std::runtime_error("Target index is too large for specified nqbits!");
    }
    if (control_ >= nqubit) {
        throw std::runtime_error("Control index is too large for specified nqbits!");
    }

    size_t nbasis = std::pow(2, nqubit);
    SparseMatrix Spmat = SparseMatrix();

    if (target_ == control_) {
        // single-qubit case
        // Iterate over the gate matrix elements
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 2; j++) {
                auto op_i_j = gate_[i][j];
                // Consider only non-zero elements (within a tolerance)
                if (std::abs(op_i_j) > 1.0e-14) {
                    // Iterate over all basis states
                    for (size_t I = 0; I < nbasis; I++) {
                        QubitBasis basis_I = QubitBasis(I);
                        // Check if the target qubit is in the correct state
                        if (basis_I.get_bit(target_) == j) {
                            QubitBasis basis_J = basis_I;
                            // Set the target qubit to the new state
                            basis_J.set_bit(target_, i);
                            // Set the corresponding element in the sparse matrix
                            Spmat.set_element(basis_J.index(), basis_I.index(), op_i_j);
                        }
                    }
                }
            }
        }
    } else {
        // two-qubit gate
        for (size_t i = 0; i < 4; i++) {
            for (size_t j = 0; j < 4; j++) {
                auto op_i_j = gate_[i][j];
                if (std::abs(op_i_j) > 1.0e-14) {
                    for (size_t I = 0; I < nbasis; I++) {
                        QubitBasis basis_I = QubitBasis(I);
                        size_t target_bit = basis_I.get_bit(target_);
                        size_t control_bit = basis_I.get_bit(control_);

                        // Combine the target and control bits into a 2-bit state
                        size_t combined_state = (control_bit << 1) | target_bit;

                        // Check if the combined state matches the gate matrix column index
                        if (combined_state == j) {
                            QubitBasis basis_J = basis_I;

                            // Extract the new states for control and target bits
                            size_t new_control_bit = (i >> 1) & 1;
                            size_t new_target_bit = i & 1;

                            // Set the target and control qubits to the new states
                            basis_J.set_bit(target_, new_target_bit);
                            basis_J.set_bit(control_, new_control_bit);

                            Spmat.set_element(basis_J.index(), basis_I.index(), op_i_j);
                        }
                    }
                }
            }
        }
    }
    return Spmat;
}

std::string Gate::gate_id() const { return label_; }

GateType Gate::gate_type() const { return type_; }

bool Gate::has_parameter() const { return parameter_.has_value(); }

std::optional<double> Gate::parameter() const {
    return has_parameter() ? std::make_optional(parameter_.value().first) : std::nullopt;
}

bool Gate::minus_parameter_on_adjoint() const {
    return has_parameter() ? parameter_.value().second : false;
}

std::string Gate::str() const {
    if (target_ == control_) {
        return fmt::format("{}{}", label_, target_);
    }
    return fmt::format("{}{}_{}", label_, target_, control_);
}

std::string Gate::repr() const {
    std::string s =
        fmt::format("{} gate, target qubit:{}, contol qubit:{}\n", label_, target_, control_);
    const std::vector<size_t>& index = (nqubits() == 1 ? index1 : index2);
    for (const auto& i : index) {
        for (const auto& j : index) {
            s += fmt::format("  {:+f} {:+f} i", std::real(gate_[i][j]), std::imag(gate_[i][j]));
        }
        s += '\n';
    }
    return s;
}

size_t Gate::nqubits() const { return (target_ == control_ ? 1 : 2); }

Gate Gate::adjoint() const {
    // test if the gate is self-adjoint and make the adjoint gate (just in case)
    std::complex<double> adj_gate[4][4];
    bool self_adjoint = true;
    for (const auto& i : index2) {
        for (const auto& j : index2) {
            adj_gate[j][i] = std::conj(gate_[i][j]);
            // test for self-adjointness
            if (std::norm(adj_gate[j][i] - gate_[j][i]) > 1.0e-12) {
                self_adjoint = false;
            }
        }
    }

    // If the gate has a parameter, then we return the gate with the appropriate parameter
    // Note that gates that have a parameter and are self-adjoint are not affected by the adjoint
    // operation. This happens for the A gate. So both cases are handled here.
    if (has_parameter()) {
        auto parameter_info = std::make_optional(std::make_pair(
            parameter_.value().first * static_cast<double>(1 - 2 * minus_parameter_on_adjoint()),
            minus_parameter_on_adjoint()));
        return Gate(label_, target_, control_, adj_gate, parameter_info);
    }

    // if the gate is self-adjoint then we return the gate itself
    if (self_adjoint) {
        return *this;
    }

    // To facilitate circuit simplification, the adjoints of S and T are expressed
    // in terms of the R phase gate.

    if (type_ == GateType::T) {
        return Gate("R", target_, control_, adj_gate, std::make_pair(-M_PI / 4, true));
    } else if (type_ == GateType::S) {
        return Gate("R", target_, control_, adj_gate, std::make_pair(-M_PI / 2, true));
    } else {
        // check if label_ is of the form adj(x) and if it is then return Gate(x)
        if (label_.size() > 4 and label_.substr(0, 4) == "adj(" and label_.back() == ')') {
            return Gate(label_.substr(4, label_.size() - 5), target_, control_, adj_gate,
                        parameter_);
        }
        return Gate("adj(" + label_ + ")", target_, control_, adj_gate, parameter_);
    }
}

const std::vector<std::pair<size_t, size_t>>& Gate::two_qubits_basis() { return two_qubits_basis_; }

bool operator==(const Gate& lhs, const Gate& rhs) {
    if (lhs.gate_id() != rhs.gate_id()) {
        return false;
    }
    if (lhs.target() != rhs.target()) {
        return false;
    }
    if (lhs.control() != rhs.control()) {
        return false;
    }
    // check if both gates have parameters
    if (lhs.has_parameter() and rhs.has_parameter()) {
        // check if both gates have the same parameter value
        return std::fabs(lhs.parameter().value() - rhs.parameter().value()) < 1.0e-12;
    }
    return true;
}

bool operator!=(const Gate& lhs, const Gate& rhs) { return not(lhs == rhs); }

bool operator<(const Gate& lhs, const Gate& rhs) {
    if (lhs.gate_id() != rhs.gate_id()) {
        return lhs.gate_id() < rhs.gate_id();
    }
    if (lhs.target() != rhs.target()) {
        return lhs.target() < rhs.target();
    }
    if (lhs.control() != rhs.control()) {
        return lhs.control() < rhs.control();
    }
    // check if both gates have parameters
    if (lhs.has_parameter() and rhs.has_parameter()) {
        // check if both gates have the same parameter value
        return lhs.parameter().value() < rhs.parameter().value();
    }
    return false;
}

GateType Gate::mapLabelToType(const std::string& label_) {
    static const std::map<std::string, GateType> labelToType = {
        {"X", GateType::X},           {"Y", GateType::Y},       {"Z", GateType::Z},
        {"H", GateType::H},           {"R", GateType::R},       {"Rx", GateType::Rx},
        {"Ry", GateType::Ry},         {"Rz", GateType::Rz},     {"V", GateType::V},
        {"S", GateType::S},           {"T", GateType::T},       {"I", GateType::I},
        {"A", GateType::A},           {"CNOT", GateType::cX},   {"cX", GateType::cX},
        {"aCNOT", GateType::acX},     {"acX", GateType::acX},   {"cY", GateType::cY},
        {"cZ", GateType::cZ},         {"cR", GateType::cR},     {"cV", GateType::cV},
        {"cRz", GateType::cRz},       {"SWAP", GateType::SWAP}, {"adj(V)", GateType::adjV},
        {"adj(cV)", GateType::adjcV},
    };

    auto it = labelToType.find(label_);
    return it != labelToType.end() ? it->second : GateType::Undefined;
}

// Define a custom hash function for pairs of GateType
struct pair_hash {
    template <class T1, class T2> std::size_t operator()(const std::pair<T1, T2>& pair) const {
        auto hash1 = std::hash<T1>{}(pair.first);
        auto hash2 = std::hash<T2>{}(pair.second);
        return hash1 ^ hash2; // Combine the two hash values
    }
};

// Custom equality to treat pairs as the same regardless of order
struct pair_equal {
    template <class T1>
    bool operator()(const std::pair<T1, T1>& lhs, const std::pair<T1, T1>& rhs) const {
        return (lhs.first == rhs.first && lhs.second == rhs.second) ||
               (lhs.first == rhs.second && lhs.second == rhs.first);
    }
};

const std::unordered_set<std::pair<GateType, GateType>, pair_hash, pair_equal>
    pairs_of_commuting_1qubit_gates = {
        {GateType::X, GateType::X},      {GateType::Rx, GateType::X},
        {GateType::V, GateType::X},      {GateType::adjV, GateType::X},
        {GateType::Y, GateType::Y},      {GateType::Ry, GateType::Y},
        {GateType::Z, GateType::Z},      {GateType::S, GateType::Z},
        {GateType::T, GateType::Z},      {GateType::Rz, GateType::Z},
        {GateType::R, GateType::Z},      {GateType::H, GateType::H},
        {GateType::S, GateType::S},      {GateType::S, GateType::T},
        {GateType::Rz, GateType::S},     {GateType::R, GateType::S},
        {GateType::T, GateType::T},      {GateType::Rz, GateType::T},
        {GateType::R, GateType::T},      {GateType::Rx, GateType::Rx},
        {GateType::Rx, GateType::V},     {GateType::adjV, GateType::Rx},
        {GateType::Ry, GateType::Ry},    {GateType::Rz, GateType::Rz},
        {GateType::R, GateType::Rz},     {GateType::R, GateType::R},
        {GateType::V, GateType::V},      {GateType::adjV, GateType::V},
        {GateType::adjV, GateType::adjV}};

const std::unordered_set<std::pair<GateType, GateType>, pair_hash, pair_equal> V_adjV = {
    {GateType::adjV, GateType::V}};

const std::unordered_set<GateType> diagonal_1qubit_gates = {GateType::T, GateType::S, GateType::Z,
                                                            GateType::Rz, GateType::R};

const std::unordered_set<GateType> phase_1qubit_gates = {GateType::T, GateType::S, GateType::Z,
                                                         GateType::R};

const std::unordered_map<GateType, GateType> controlled_2qubit_to_1qubit_gate = {
    {GateType::cX, GateType::X}, {GateType::acX, GateType::X},      {GateType::cY, GateType::Y},
    {GateType::cZ, GateType::Z}, {GateType::cRz, GateType::Rz},     {GateType::cR, GateType::R},
    {GateType::cV, GateType::V}, {GateType::adjcV, GateType::adjV},
};

const std::unordered_set<GateType> symmetrical_2qubit_gates = {GateType::cZ, GateType::cR,
                                                               GateType::SWAP};

std::pair<bool, int> evaluate_gate_interaction(const Gate& gate1, const Gate& gate2) {

    std::set<size_t> gate1_qubits = {gate1.target(), gate1.control()};
    std::set<size_t> gate2_qubits = {gate2.target(), gate2.control()};

    // find number of common qubits
    int commonQubitCount = 0;
    for (size_t q : gate1_qubits) {
        if (gate2_qubits.find(q) != gate2_qubits.end()) {
            ++commonQubitCount;
        }
    }

    if (commonQubitCount == 0) {
        return {true, 0}; // Disjoint qubits
    }

    int num_qubits_gate1 = gate1_qubits.size();
    int num_qubits_gate2 = gate2_qubits.size();
    int product_nqubits = num_qubits_gate1 * num_qubits_gate2;

    if (product_nqubits == 1) {
        if (phase_1qubit_gates.find(gate1.gate_type()) != phase_1qubit_gates.end() &&
            phase_1qubit_gates.find(gate2.gate_type()) != phase_1qubit_gates.end()) {
            if (gate1.gate_type() == gate2.gate_type()) {
                return {true, 1};
            } else {
                return {true, 3};
            }
        }
        std::pair<GateType, GateType> pairGateType =
            std::make_pair(gate1.gate_type(), gate2.gate_type());
        return {pairs_of_commuting_1qubit_gates.find(pairGateType) !=
                    pairs_of_commuting_1qubit_gates.end(),
                (gate1.gate_type() == gate2.gate_type()) +
                    4 * (V_adjV.find(pairGateType) != V_adjV.end())};
    }

    if (product_nqubits == 2) {
        if (gate1.gate_type() == GateType::SWAP || gate2.gate_type() == GateType::SWAP ||
            gate1.gate_type() == GateType::A || gate2.gate_type() == GateType::A) {
            return {false, 0};
        }
        const Gate& single_qubit_gate = (num_qubits_gate1 == 1) ? gate1 : gate2;
        const Gate& two_qubit_gate = (num_qubits_gate1 == 1) ? gate2 : gate1;

        if (single_qubit_gate.target() == two_qubit_gate.target()) {
            std::pair<GateType, GateType> pairGateType =
                std::make_pair(single_qubit_gate.gate_type(),
                               controlled_2qubit_to_1qubit_gate.at(two_qubit_gate.gate_type()));
            return {pairs_of_commuting_1qubit_gates.find(pairGateType) !=
                        pairs_of_commuting_1qubit_gates.end(),
                    0};
        }
        return {diagonal_1qubit_gates.find(single_qubit_gate.gate_type()) !=
                    diagonal_1qubit_gates.end(),
                0};
    }

    if (product_nqubits == 4) {
        if (commonQubitCount == 1) {
            if (gate1.gate_type() == GateType::SWAP || gate2.gate_type() == GateType::SWAP ||
                gate1.gate_type() == GateType::A || gate2.gate_type() == GateType::A) {
                return {false, 0};
            }
            if (gate1.control() == gate2.control()) {
                return {true, 0};
            }
            if (gate1.target() == gate2.target()) {
                std::pair<GateType, GateType> pairGateType =
                    std::make_pair(controlled_2qubit_to_1qubit_gate.at(gate1.gate_type()),
                                   controlled_2qubit_to_1qubit_gate.at(gate2.gate_type()));
                return {pairs_of_commuting_1qubit_gates.find(pairGateType) !=
                            pairs_of_commuting_1qubit_gates.end(),
                        0};
            }
            if (gate1.target() == gate2.control()) {
                return {diagonal_1qubit_gates.find(controlled_2qubit_to_1qubit_gate.at(
                            gate1.gate_type())) != diagonal_1qubit_gates.end(),
                        0};
            }
            return {diagonal_1qubit_gates.find(controlled_2qubit_to_1qubit_gate.at(
                        gate2.gate_type())) != diagonal_1qubit_gates.end(),
                    0};
        } else {
            if (symmetrical_2qubit_gates.find(gate1.gate_type()) !=
                    symmetrical_2qubit_gates.end() &&
                symmetrical_2qubit_gates.find(gate2.gate_type()) !=
                    symmetrical_2qubit_gates.end()) {
                return {true, 2 * (gate1.gate_type() == gate2.gate_type())};
            }
            if (gate1.gate_type() == GateType::SWAP || gate2.gate_type() == GateType::SWAP) {
                return {false, 0};
            }
            if (gate1.gate_type() == GateType::A || gate2.gate_type() == GateType::A) {
                if (symmetrical_2qubit_gates.find(gate1.gate_type()) !=
                        symmetrical_2qubit_gates.end() ||
                    symmetrical_2qubit_gates.find(gate2.gate_type()) !=
                        symmetrical_2qubit_gates.end()) {
                    return {true, 0};
                } else {
                    return {false, 0};
                }
            }
            if (gate1.target() == gate2.target()) {
                std::pair<GateType, GateType> pairGateType =
                    std::make_pair(controlled_2qubit_to_1qubit_gate.at(gate1.gate_type()),
                                   controlled_2qubit_to_1qubit_gate.at(gate2.gate_type()));
                return {pairs_of_commuting_1qubit_gates.find(pairGateType) !=
                            pairs_of_commuting_1qubit_gates.end(),
                        (gate1.gate_type() == gate2.gate_type()) +
                            4 * (V_adjV.find(pairGateType) != V_adjV.end())};
            }
            return {diagonal_1qubit_gates.find(controlled_2qubit_to_1qubit_gate.at(
                        gate1.gate_type())) != diagonal_1qubit_gates.end() &&
                        diagonal_1qubit_gates.find(controlled_2qubit_to_1qubit_gate.at(
                            gate2.gate_type())) != diagonal_1qubit_gates.end(),
                    0};
        }
    }

    return {false, 0}; // Default return
};
