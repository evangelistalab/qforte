#include "helpers.h"
#include "quantum_gate.h"
#include "quantum_circuit.h"
#include "quantum_operator.h"
#include "sq_operator.h"
#include "sq_op_pool.h"
#include "quantum_op_pool.h"

#include <stdexcept>
#include <algorithm>

void QuantumOpPool::add_term(std::complex<double> coeff, const QuantumOperator& sq_op ){
    terms_.push_back(std::make_pair(coeff, sq_op));
}

void QuantumOpPool::set_coeffs(const std::vector<std::complex<double>>& new_coeffs){
    if(new_coeffs.size() != terms_.size()){
        throw std::invalid_argument( "Number of new coeficients for quantum operator must equal." );
    }
    for (size_t l = 0; l < new_coeffs.size(); l++){
        terms_[l].first = new_coeffs[l];
    }
}

void QuantumOpPool::set_terms(std::vector<std::pair<std::complex<double>, QuantumOperator>>& new_terms) {
    // TODO: consider clearing terms_ when this fuction is called
    for(const auto& term : new_terms){
        terms_.push_back(term);
    }
}

const std::vector<std::pair<std::complex<double>, QuantumOperator>>& QuantumOpPool::terms() const{
    return terms_;
}

// const std::vector< QuantumOperator>& QuantumOpPool::terms() const{
//     return terms_;
// }

void QuantumOpPool::set_orb_spaces(const std::vector<int>& ref){
    int norb = ref.size();
    if (norb%2 == 0){
        norb = static_cast<int>(norb/2);
    } else {
        throw std::invalid_argument("QForte does not yet support systems with an odd number of spin orbitals.");
    }

    nocc_ = 0;
    for (const auto& occupancy : ref){
        nocc_ += occupancy;
    }

    if (nocc_%2 == 0){
        nocc_ = static_cast<int>(nocc_/2);
    } else {
        throw std::invalid_argument("QForte does not yet support systems with an odd number of occupied spin orbitals.");
    }

    nvir_ = static_cast<int>(norb - nocc_);
}


void QuantumOpPool::join_op_from_right(const QuantumOperator& q_op){
    for (auto& term : terms_) {
        term.second.join_operator(q_op, false);
        term.second.simplify();
    }
}

void QuantumOpPool::join_op_from_left(const QuantumOperator& q_op){
    std::vector<std::pair<std::complex<double>, QuantumOperator>> temp_terms;
    for (const auto& term : terms_) {
        QuantumOperator temp_op;
        temp_op.add_op(q_op);
        temp_op.join_operator(term.second, false);
        temp_op.simplify();
        temp_terms.push_back(std::make_pair(term.first, temp_op));
    }
    terms_ = std::move(temp_terms);
}

void QuantumOpPool::join_as_comutator(const QuantumOperator& q_op){
    std::vector<std::pair<std::complex<double>, QuantumOperator>> temp_terms;
    for (const auto& term : terms_) {
        // build HAm
        QuantumOperator HAm;
        HAm.add_op(q_op);
        HAm.join_operator(term.second, false);
        // HAm.simplify();

        // build -AmH
        QuantumOperator AmH;
        AmH.add_op(term.second);
        AmH.join_operator(q_op, false);
        AmH.mult_coeffs(-1.0);
        // AmH.simplify();

        HAm.add_op(AmH);
        HAm.simplify();
        temp_terms.push_back(std::make_pair(term.first, HAm));
    }
    terms_ = std::move(temp_terms);
}

void QuantumOpPool::fill_pool(std::string pool_type){
    if(pool_type == "test"){
        QuantumOperator A;
        QuantumCircuit a1;
        QuantumCircuit a2;
        a1.add_gate(make_gate("X", 0, 0));
        a1.add_gate(make_gate("Y", 1, 1));
        a1.add_gate(make_gate("Z", 2, 2));
        a2.add_gate(make_gate("X", 2, 2));
        a2.add_gate(make_gate("Y", 1, 1));
        a2.add_gate(make_gate("Z", 0, 0));
        A.add_term(+0.5, a1);
        A.add_term(-0.5, a2);
        add_term(4.11, A);
    } else {
        throw std::invalid_argument( "Invalid pool_type specified." );
    }
}

std::string QuantumOpPool::str() const{
    std::vector<std::string> s;
    s.push_back("");
    int counter = 0;
    for (const auto& term : terms_) {
        s.push_back("----->");
        s.push_back(std::to_string(counter));
        s.push_back("<-----\n");
        s.push_back(to_string(term.first));
        s.push_back("[\n");
        s.push_back(term.second.str());
        s.push_back("\n");
        s.push_back("]\n\n");
        counter++;
    }
    return join(s, " ");
}
