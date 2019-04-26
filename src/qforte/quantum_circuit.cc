#include "quantum_gate.h"

#include "quantum_circuit.h"

void QuantumCircuit::set_parameters(const std::vector<double>& params) {
    // need a loop over only gates in state preparation circuit that
    // have a parameter dependance (if gate_id == Rx, Ry, or Rz)
    // TODO: make a indexing funciton using a map (Nick)
    size_t param_idx = 0;
    for (auto& gate : gates_) {
        std::string gate_id = gate.gate_id();
        if (gate_id == "Rz") {
            size_t target_qubit = gate.target();
            gate = make_gate(gate_id, target_qubit, target_qubit, params[param_idx]);
            param_idx++;
        }
    }
}

void QuantumCircuit::add_circuit(const QuantumCircuit& circ) {
    for (const auto gate : circ.gates()) {
        gates_.push_back(gate);
    }
}

QuantumCircuit QuantumCircuit::adjoint() {
    QuantumCircuit qcirc_adjoint;
    for (auto& gate : gates_) {
        qcirc_adjoint.add_gate(gate.adjoint());
    }
    std::reverse(std::begin(qcirc_adjoint.gates_), std::end(qcirc_adjoint.gates_));
    return qcirc_adjoint;
}

std::vector<std::string> QuantumCircuit::str() const {
    std::vector<std::string> s;
    for (const auto& gate : gates_) {
        s.push_back(gate.str());
    }
    return s;
}

// std::vector<double> QuantumCircuit::get_parameters() {
//     // need a loop over only gates in state preparation circuit that
//     // have a parameter dependance (if gate_id == Rx, Ry, or Rz)
//     // TODO: make a indexing funciton using a map (Nick)
//     size_t param_idx = 0;
//     std::vector<double> params
//     for (auto& gate : gates_) {
//         std::string gate_id = gate.gate_id();
//         if (gate_id == "Rz") {
//
//             double param = gate.gate()[][];
//             gate = make_gate(gate_id, target_qubit, target_qubit, params[param_idx]);
//             param_idx++;
//         }
//     }
// }
