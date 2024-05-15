#include <algorithm>
#include <numeric>

#include "helpers.h"
#include "gate.h"
#include "circuit.h"
#include "computer.h"
#include "sparse_tensor.h"

void Circuit::set_parameters(const std::vector<double>& params) {
    size_t param_idx = 0;
    auto param_size = params.size();
    for (auto& gate : gates_) {
        if (gate.has_parameter()) {
            // check if the parameter is close to the current value. If so, don't update it
            if ((std::abs(gate.parameter().value() - params[param_idx]) > 1e-12) and
                (param_idx < param_size)) {
                gate = make_gate(gate.gate_id(), gate.target(), gate.control(), params[param_idx]);
            }
            param_idx++;
        }
    }
    if (param_idx != param_size) {
        throw std::runtime_error("Circuit::set_parameters: number of parameters does not match "
                                 "number of gates with parameters");
    }
}

void Circuit::set_parameter(size_t pos, double param) {
    if (pos >= gates_.size()) {
        throw std::runtime_error("Circuit::set_parameter: position out of range");
    }
    if (!gates_[pos].has_parameter()) {
        throw std::runtime_error("Circuit::set_parameter: gate does not have a parameter");
    }
    gates_[pos] =
        make_gate(gates_[pos].gate_id(), gates_[pos].target(), gates_[pos].control(), param);
}

std::vector<double> Circuit::get_parameters() const {
    std::vector<double> params;
    for (const auto& gate : gates_) {
        if (gate.has_parameter()) {
            params.push_back(gate.parameter().value());
        }
    }
    return params;
}

void Circuit::insert_gate(size_t pos, const Gate& gate) {
    if (pos > gates_.size()) {
        throw std::runtime_error("Circuit::insert_gate: position out of range");
    }
    gates_.insert(gates_.begin() + pos, gate);
}

void Circuit::remove_gate(size_t pos) {
    if (pos >= gates_.size()) {
        throw std::runtime_error("Circuit::remove_gate: position out of range");
    }
    gates_.erase(gates_.begin() + pos);
}

void Circuit::replace_gate(size_t pos, const Gate& gate) {
    if (pos >= gates_.size()) {
        throw std::runtime_error("Circuit::replace_gate: position out of range");
    }
    gates_[pos] = gate;
}

void Circuit::swap_gates(size_t pos1, size_t pos2) {
    if (pos1 >= gates_.size() || pos2 >= gates_.size()) {
        throw std::runtime_error("Circuit::swap_gates: position out of range");
    }
    std::swap(gates_[pos1], gates_[pos2]);
}

void Circuit::add_circuit(const Circuit& circ) {
    for (const auto gate : circ.gates()) {
        gates_.push_back(gate);
    }
}

void Circuit::insert_circuit(size_t pos, const Circuit& circ) {
    if (pos > gates_.size()) {
        throw std::runtime_error("Circuit::insert_circuit: position out of range");
    }
    gates_.insert(gates_.begin() + pos, circ.gates().begin(), circ.gates().end());
}

void Circuit::remove_gates(size_t pos1, size_t pos2) {
    if (pos1 >= gates_.size() || pos2 >= gates_.size()) {
        throw std::runtime_error("Circuit::remove_gates: position out of range");
    }
    gates_.erase(gates_.begin() + pos1, gates_.begin() + pos2);
}

Circuit Circuit::adjoint() {
    Circuit qcirc_adjoint;
    for (auto& gate : gates_) {
        qcirc_adjoint.add_gate(gate.adjoint());
    }
    std::reverse(std::begin(qcirc_adjoint.gates_), std::end(qcirc_adjoint.gates_));
    return qcirc_adjoint;
}

std::complex<double> Circuit::canonicalize_pauli_circuit() {
    if (gates_.size() == 0) {
        // If there are no gates, there's nothing to order.
        return 1.0;
    }
    if (!is_pauli()) {
        throw std::runtime_error(
            "Circuit::canonicalize_pauli_circuit is undefined for circuits with gates other "
            "than "
            "X, Y, or Z");
    }
    // using namespace std::complex_literals;
    std::complex<double> onei(0.0, 1.0);
    std::map<std::pair<std::string, std::string>, std::pair<std::complex<double>, std::string>> m =
        {{std::make_pair("X", "Y"), std::make_pair(onei, "Z")},
         {std::make_pair("X", "Z"), std::make_pair(-onei, "Y")},
         {std::make_pair("Y", "X"), std::make_pair(-onei, "Z")},
         {std::make_pair("Y", "Z"), std::make_pair(onei, "X")},
         {std::make_pair("Z", "X"), std::make_pair(onei, "Y")},
         {std::make_pair("Z", "Y"), std::make_pair(-onei, "X")},
         {std::make_pair("X", "X"), std::make_pair(1.0, "I")},
         {std::make_pair("Y", "Y"), std::make_pair(1.0, "I")},
         {std::make_pair("Z", "Z"), std::make_pair(1.0, "I")},
         {std::make_pair("I", "X"), std::make_pair(1.0, "X")},
         {std::make_pair("I", "Y"), std::make_pair(1.0, "Y")},
         {std::make_pair("I", "Z"), std::make_pair(1.0, "Z")}};

    // Apply gate commutation to sort gates from those acting on smallest-index qubit to
    // largest.
    std::stable_sort(gates_.begin(), gates_.end(),
                     [&](const Gate& a, const Gate& b) { return (a.target() < b.target()); });

    int n_gates = gates_.size();
    Circuit simplified_circ;
    std::complex<double> coeff = 1.0;
    std::string s;
    bool first_gate_for_qubit = true;

    for (int i = 0; i < n_gates; i++) {
        if (first_gate_for_qubit) {
            s = gates_[i].gate_id();
        }
        if (i + 1 != n_gates && gates_[i].target() == gates_[i + 1].target()) {
            // The upcoming gate also acts on this qubit, and it exists. Time to update s.
            const auto& qubit_update = m[std::make_pair(s, gates_[i + 1].gate_id())];
            coeff *= qubit_update.first;
            s = qubit_update.second;
            first_gate_for_qubit = false;
        } else {
            // The upcoming gate does not act on this qubit or doesn't exist.
            // Let's add the current qubit, if it's non-trivial.
            if (s != "I") {
                simplified_circ.add_gate(make_gate(s, gates_[i].target(), gates_[i].target()));
            }
            first_gate_for_qubit = true;
        }
    }

    // copy simplified terms_
    // maybe use copy, not sure move is faster here?
    gates_ = std::move(simplified_circ.gates());
    return coeff;
}

int Circuit::get_num_cnots() const {
    int n_cnots = 0;
    for (const auto& gate : gates_) {
        if (gate.gate_id() == "CNOT" || gate.gate_id() == "cX" || gate.gate_id() == "aCNOT" ||
            gate.gate_id() == "acX") {
            n_cnots++;
        } else if (gate.gate_id() == "A" || gate.gate_id() == "SWAP") {
            n_cnots += 3;
        }
    }
    return n_cnots;
}

std::string Circuit::str() const {
    std::vector<std::string> s;
    for (const auto& gate : gates_) {
        s.push_back(gate.str());
    }
    std::reverse(s.begin(), s.end());
    return "[" + join(s, " ") + "]";
}

size_t Circuit::num_qubits() const {
    size_t max = 0;
    for (const auto& gate : gates_) {
        max = std::max({max, gate.target() + 1, gate.control() + 1});
    }
    return max;
}

const SparseMatrix Circuit::sparse_matrix(size_t nqubit) const {
    size_t ngates = gates_.size();
    if (ngates == 0) {
        SparseMatrix Rmat = SparseMatrix();
        size_t nbasis = std::pow(2, nqubit);
        Rmat.make_identity(nbasis);
        return Rmat;
    }

    SparseMatrix Rmat = gates_[0].sparse_matrix(nqubit);

    for (size_t i = 1; i < ngates; i++) {
        SparseMatrix Lmat = gates_[i].sparse_matrix(nqubit);
        Rmat.left_multiply(Lmat);
    }
    return Rmat;
}

bool Circuit::is_pauli() const {
    for (const auto& gate : gates_) {
        const auto& id = gate.gate_id();
        if (id != "X" && id != "Y" && id != "Z") {
            return false;
        }
    }
    return true;
}

double Circuit::get_phase_gate_parameter(const Gate& gate) {
    std::unordered_map<GateType, double> gate_parameters = {
        {GateType::T, M_PI / 4}, {GateType::S, M_PI / 2}, {GateType::Z, M_PI}};

    if (gate.has_parameter()) {
        return gate.parameter().value();
    } else {
        auto it = gate_parameters.find(gate.gate_type());
        if (it != gate_parameters.end()) {
            return it->second;
        } else {
            throw std::invalid_argument("Unknown single-qubit phase gate encountered: " +
                                        gate.gate_id());
        }
    }
}

void Circuit::simplify() {

    const std::unordered_set<GateType> involutory_gates = {
        GateType::X,  GateType::Y,   GateType::Z, GateType::cX,  GateType::cY,
        GateType::cZ, GateType::acX, GateType::H, GateType::SWAP};

    const std::unordered_set<GateType> parametrized_gates = {
        GateType::Rx, GateType::Ry, GateType::Rz, GateType::R, GateType::cRz, GateType::cR};

    const std::unordered_set<GateType> square_root_gates = {
        GateType::T, GateType::S, GateType::V, GateType::cV, GateType::adjV, GateType::adjcV};

    const std::unordered_map<GateType, std::string> simplify_square_root_gates = {
        {GateType::T, "S"},   {GateType::S, "Z"},    {GateType::V, "X"},
        {GateType::cV, "cX"}, {GateType::adjV, "X"}, {GateType::adjcV, "cX"}};

    std::vector<size_t> gate_indices_to_remove;

    for (size_t pos1 = 0; pos1 < gates_.size(); pos1++) {
        if (std::find(gate_indices_to_remove.begin(), gate_indices_to_remove.end(), pos1) !=
            gate_indices_to_remove.end()) {
            continue;
        }
        Gate gate1 = gates_[pos1];
        for (size_t pos2 = pos1 + 1; pos2 < gates_.size(); pos2++) {
            if (std::find(gate_indices_to_remove.begin(), gate_indices_to_remove.end(), pos2) !=
                gate_indices_to_remove.end()) {
                continue;
            }
            Gate gate2 = gates_[pos2];
            bool commute;
            int simplification_case;
            std::tie(commute, simplification_case) = evaluate_gate_interaction(gate1, gate2);
            if (commute) {
                if (simplification_case == 0) {
                    continue;
                }
                if (simplification_case == 1) {
                    if (involutory_gates.find(gate1.gate_type()) != involutory_gates.end()) {
                        gate_indices_to_remove.push_back(pos1);
                        gate_indices_to_remove.push_back(pos2);
                        break;
                    }
                    if (parametrized_gates.find(gate1.gate_type()) != parametrized_gates.end()) {
                        gate_indices_to_remove.push_back(pos1);
                        gates_[pos2] = make_gate(gates_[pos2].gate_id(), gates_[pos2].target(),
                                                 gates_[pos2].control(),
                                                 *gate1.parameter() + *gate2.parameter());
                        break;
                    }
                    if (square_root_gates.find(gate1.gate_type()) != square_root_gates.end()) {
                        gate_indices_to_remove.push_back(pos1);
                        gates_[pos2] = make_gate(simplify_square_root_gates.at(gate2.gate_type()),
                                                 gates_[pos2].target(), gates_[pos2].control());
                        break;
                    }
                }
                if (simplification_case == 2) {
                    if (involutory_gates.find(gate1.gate_type()) != involutory_gates.end()) {
                        gate_indices_to_remove.push_back(pos1);
                        gate_indices_to_remove.push_back(pos2);
                        break;
                    }
                    if (phase_1qubit_gates.find(controlled_2qubit_to_1qubit_gate.at(
                            gate1.gate_type())) != phase_1qubit_gates.end()) {
                        gate_indices_to_remove.push_back(pos1);
                        gates_[pos2] = make_gate(gates_[pos2].gate_id(), gates_[pos2].target(),
                                                 gates_[pos2].control(),
                                                 *gate1.parameter() + *gate2.parameter());
                        break;
                    }
                }
                if (simplification_case == 3) {
                    gate_indices_to_remove.push_back(pos1);
                    gates_[pos2] = make_gate("R", gates_[pos2].target(), gates_[pos2].control(),
                                             get_phase_gate_parameter(gate1) +
                                                 get_phase_gate_parameter(gate2));
                    break;
                }
                if (simplification_case == 4) {
                    gate_indices_to_remove.push_back(pos1);
                    gate_indices_to_remove.push_back(pos2);
                    break;
                }
            } else {
                break;
            }
        }
    }
    std::sort(gate_indices_to_remove.begin(), gate_indices_to_remove.end());
    std::reverse(gate_indices_to_remove.begin(), gate_indices_to_remove.end());
    for (size_t pos : gate_indices_to_remove) {
        gates_.erase(gates_.begin() + pos);
    }
}

bool operator==(const Circuit& circ1, const Circuit& circ2) {
    return circ1.gates() == circ2.gates();
}

bool operator<(const Circuit& circ1, const Circuit& circ2) { return circ1.gates() < circ2.gates(); }
