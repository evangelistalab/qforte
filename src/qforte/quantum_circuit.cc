#include <algorithm>

#include "helpers.h"
#include "gate.h"
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

std::complex<double> QuantumCircuit::canonicalize_pauli_circuit() {
    if (gates_.size()==0){
        // If there are no gates, there's nothing to order.
        return 1.0;
    }
    for (const auto& gate: gates_) {
        const auto& id = gate.gate_id();
        if (id != "X" && id != "Y" && id != "Z") {
            throw ("QuantumCircuit::canonicalize_pauli_circuit is undefined for circuits with gates other than X, Y, or Z");
        }
    }
    //using namespace std::complex_literals;
    std::complex<double> onei(0.0, 1.0);
    std::map<
        std::pair<std::string,std::string> ,
        std::pair<std::complex<double>,std::string>
        > m = {
        {std::make_pair("X", "Y"), std::make_pair( onei, "Z")},
        {std::make_pair("X", "Z"), std::make_pair(-onei, "Y")},
        {std::make_pair("Y", "X"), std::make_pair(-onei, "Z")},
        {std::make_pair("Y", "Z"), std::make_pair( onei, "X")},
        {std::make_pair("Z", "X"), std::make_pair( onei, "Y")},
        {std::make_pair("Z", "Y"), std::make_pair(-onei, "X")},
        {std::make_pair("X", "X"), std::make_pair( 1.0,  "I")},
        {std::make_pair("Y", "Y"), std::make_pair( 1.0,  "I")},
        {std::make_pair("Z", "Z"), std::make_pair( 1.0,  "I")},
        {std::make_pair("I", "X"), std::make_pair( 1.0,  "X")},
        {std::make_pair("I", "Y"), std::make_pair( 1.0,  "Y")},
        {std::make_pair("I", "Z"), std::make_pair( 1.0,  "Z")}
    };

    // Apply gate commutation to sort gates from those acting on smallest-index qubit to largest.
    std::stable_sort(gates_.begin(), gates_.end(),
        [&](const Gate& a, const Gate& b) {
            return (a.target() < b.target());
        }
    );

    int n_gates = gates_.size();
    QuantumCircuit simplified_circ;
    std::complex<double> coeff = 1.0;
    std::string s;
    bool first_gate_for_qubit = true;

    for (int i=0; i < n_gates; i++) {
        if (first_gate_for_qubit) {
            s = gates_[i].gate_id();
        }
        if(gates_[i].target() == gates_[i+1].target() && i + 1 != n_gates) {
            // The upcoming gate also acts on this qubit, and it exists. Time to update s.
            const auto& qubit_update = m[std::make_pair(s, gates_[i+1].gate_id())];
            coeff *= qubit_update.first;
            s = qubit_update.second;
            first_gate_for_qubit = false;
        } else {
            // The upcoming gate does not act on this qubit or doesn't exist.
            // Let's add the current qubit, if it's non-trivial.
            if (s != "I") {
                simplified_circ.add_gate(
                    make_gate(s, gates_[i].target(), gates_[i].target())
                );
            }
            first_gate_for_qubit = true;
        }
    }

    // copy simplified terms_
    // maybe use copy, not sure move is faster here?
    gates_ = std::move(simplified_circ.gates());
    return coeff;
}

int QuantumCircuit::get_num_cnots() const {
    int n_cnots = 0;
    for (const auto& gate : gates_) {
        if(gate.gate_id() == "CNOT" || gate.gate_id() == "cX"){
            n_cnots++;
        }
    }
    return n_cnots;
}

std::string QuantumCircuit::str() const {
    std::vector<std::string> s;
    for (const auto& gate : gates_) {
        s.push_back(gate.str());
    }
    return "[" + join(s, " ") + "]";
}

size_t QuantumCircuit::num_qubits() const {
    size_t max = 0;
    for (const auto& gate: gates_) {
        max = std::max({max, gate.target() + 1, gate.control() + 1});
    }
    return max;
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

bool operator==(const QuantumCircuit& qc1, const QuantumCircuit& qc2)  {
    if(qc1.gates().size() == qc2.gates().size()){
        for (int k=0; k<qc1.gates().size(); k++){
            if (qc1.gates()[k].gate_id() != qc2.gates()[k].gate_id()){
                return false;
            } else if (qc1.gates()[k].target() != qc2.gates()[k].target()) {
                return false;
            } else if (qc1.gates()[k].control() != qc2.gates()[k].control()) {
                return false;
            }
        }
        return true;
    } else {
        return false;
    }
}
