#include <algorithm>
#include <numeric>

#include "helpers.h"
#include "gate.h"
#include "circuit.h"
#include "computer.h"
#include "sparse_tensor.h"

void Circuit::set_parameters(const std::vector<double>& params) {
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

void Circuit::add_circuit(const Circuit& circ) {
    for (const auto gate : circ.gates()) {
        gates_.push_back(gate);
    }
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
    if (gates_.size()==0){
        // If there are no gates, there's nothing to order.
        return 1.0;
    }
    for (const auto& gate: gates_) {
        const auto& id = gate.gate_id();
        if (id != "X" && id != "Y" && id != "Z") {
            throw ("Circuit::canonicalize_pauli_circuit is undefined for circuits with gates other than X, Y, or Z");
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
    Circuit simplified_circ;
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

int Circuit::get_num_cnots() const {
    int n_cnots = 0;
    for (const auto& gate : gates_) {
        if(gate.gate_id() == "CNOT" || gate.gate_id() == "cX" ||
                gate.gate_id() == "aCNOT" || gate.gate_id() == "acX"){
            n_cnots++;
        } else if (gate.gate_id() == "A"){
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
    for (const auto& gate: gates_) {
        max = std::max({max, gate.target() + 1, gate.control() + 1});
    }
    return max;
}

const SparseMatrix Circuit::sparse_matrix(size_t nqubit) const {
    size_t ngates = gates_.size();
    if (ngates==0){
        SparseMatrix Rmat = SparseMatrix();
        size_t nbasis = std::pow(2, nqubit);
        Rmat.make_identity(nbasis);
        return Rmat;
    }

    SparseMatrix Rmat = gates_[0].sparse_matrix(nqubit);

    for(size_t i=1; i < ngates ;i++){
        SparseMatrix Lmat = gates_[i].sparse_matrix(nqubit);
        Rmat.left_multiply(Lmat);
    }
    return Rmat;
}

// std::vector<double> Circuit::get_parameters() {
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

bool operator==(const Circuit& qc1, const Circuit& qc2)  {
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
