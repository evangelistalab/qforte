#include "helpers.h"
#include "quantum_gate.h"
#include "quantum_circuit.h"
#include "quantum_operator.h"
#include "sq_operator.h"
#include "sq_op_pool.h"
#include "quantum_op_pool.h"

#include <stdexcept>
#include <algorithm>
#include <iostream>

void QuantumOpPool::add_term(std::complex<double> coeff, const QuantumOperator& sq_op ){
    terms_.push_back(std::make_pair(coeff, sq_op));
}

void QuantumOpPool::set_coeffs(const std::vector<std::complex<double>>& new_coeffs){
    if(new_coeffs.size() != terms_.size()){
        throw std::invalid_argument( "Number of new coeficients for quantum op pool must equal number of terms in pool." );
    }
    for (size_t l = 0; l < new_coeffs.size(); l++){
        terms_[l].first = new_coeffs[l];
    }
}

void QuantumOpPool::set_op_coeffs(const std::vector<std::complex<double>>& new_coeffs){
    for (size_t I = 0; I < terms_.size(); I++){
        if(new_coeffs.size() != terms_[I].second.terms().size()){
            throw std::invalid_argument( "Number of new coeficients for quantum operator must equal number of terim in operator." );
        }
        terms_[I].second.set_coeffs(new_coeffs);
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

void QuantumOpPool::square(bool upper_triangle_only){
    std::vector<std::pair<std::complex<double>, QuantumOperator>> temp_terms;
    if(upper_triangle_only){
        //consider only I -> IJ, where J > I
        for (int I=0; I<terms_.size(); I++) {
            for (int J=I; J<terms_.size(); J++) {
                QuantumOperator IJ;
                IJ.add_op(terms_[I].second);
                IJ.join_operator(terms_[J].second, false);
                IJ.simplify();
                temp_terms.push_back(
                    std::make_pair(std::conj(terms_[I].first) * terms_[J].first, IJ)
                );
            }
        }
    } else {
        //consider all I -> IJ
        for (auto& I : terms_) {
            for (auto& J : terms_) {
                QuantumOperator IJ;
                IJ.add_op(I.second);
                IJ.join_operator(J.second, false);
                IJ.simplify();
                temp_terms.push_back(
                    std::make_pair(std::conj(I.first) * J.first, IJ)
                );
            }
        }
    }
    terms_ = std::move(temp_terms);
}

void QuantumOpPool::join_op_from_right(const QuantumOperator& q_op){
    for (auto& term : terms_) {
        term.second.join_operator(q_op, false);
        term.second.simplify();
    }
}

void QuantumOpPool::join_op_from_right_lazy(const QuantumOperator& q_op){
    for (auto& term : terms_) {
        term.second.join_operator_lazy(q_op);
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

void QuantumOpPool::fill_pool(std::string pool_type, const std::vector<int>& ref){
    if(pool_type == "test"){
        // [86, 89, 101, 106, 149, 154, 166, 169]
        QuantumOperator A1;
        QuantumOperator A2;
        QuantumOperator A3;
        QuantumOperator A4;
        QuantumOperator A5;
        QuantumOperator A6;
        QuantumOperator A7;
        QuantumOperator A8;

        QuantumCircuit a1;
        QuantumCircuit a2;
        QuantumCircuit a3;
        QuantumCircuit a4;
        QuantumCircuit a5;
        QuantumCircuit a6;
        QuantumCircuit a7;
        QuantumCircuit a8;

        //[X0 X1 X2 Y3]
        a1.add_gate(make_gate("X", 0, 0));
        a1.add_gate(make_gate("X", 1, 1));
        a1.add_gate(make_gate("X", 2, 2));
        a1.add_gate(make_gate("Y", 3, 3));
        // [X0 X1 Y2 X3]
        a2.add_gate(make_gate("X", 0, 0));
        a2.add_gate(make_gate("X", 1, 1));
        a2.add_gate(make_gate("Y", 2, 2));
        a2.add_gate(make_gate("X", 3, 3));
        // [X0 Y1 X2 X3]
        a3.add_gate(make_gate("X", 0, 0));
        a3.add_gate(make_gate("Y", 1, 1));
        a3.add_gate(make_gate("X", 2, 2));
        a3.add_gate(make_gate("X", 3, 3));
        // [X0 Y1 Y2 Y3]
        a4.add_gate(make_gate("X", 0, 0));
        a4.add_gate(make_gate("Y", 1, 1));
        a4.add_gate(make_gate("Y", 2, 2));
        a4.add_gate(make_gate("Y", 3, 3));
        // [Y0 X1 X2 X3]
        a5.add_gate(make_gate("Y", 0, 0));
        a5.add_gate(make_gate("X", 1, 1));
        a5.add_gate(make_gate("X", 2, 2));
        a5.add_gate(make_gate("X", 3, 3));
        // [Y0 X1 Y2 Y3]
        a6.add_gate(make_gate("Y", 0, 0));
        a6.add_gate(make_gate("X", 1, 1));
        a6.add_gate(make_gate("Y", 2, 2));
        a6.add_gate(make_gate("Y", 3, 3));
        //[Y0 Y1 X2 Y3]
        a7.add_gate(make_gate("Y", 0, 0));
        a7.add_gate(make_gate("Y", 1, 1));
        a7.add_gate(make_gate("X", 2, 2));
        a7.add_gate(make_gate("Y", 3, 3));
        //[Y0 Y1 Y2 X3]
        a8.add_gate(make_gate("Y", 0, 0));
        a8.add_gate(make_gate("Y", 1, 1));
        a8.add_gate(make_gate("Y", 2, 2));
        a8.add_gate(make_gate("X", 3, 3));

        A1.add_term(1.0, a1);
        A2.add_term(1.0, a2);
        A3.add_term(1.0, a3);
        A4.add_term(1.0, a4);
        A5.add_term(1.0, a5);
        A6.add_term(1.0, a6);
        A7.add_term(1.0, a7);
        A8.add_term(1.0, a8);

        add_term(1.0, A1);
        add_term(1.0, A2);
        add_term(1.0, A3);
        add_term(1.0, A4);
        add_term(1.0, A5);
        add_term(1.0, A6);
        add_term(1.0, A7);
        add_term(1.0, A8);

    } else if(pool_type == "complete_qubit") {
        std::map<std::string, std::string> paulis = {{"0","I"}, {"1","X"}, {"2","Y"}, {"3","Z"}};
        int nterms = static_cast<int>(std::pow(4, ref.size()));

        for(int I=0; I<nterms; I++){
            QuantumOperator AI;
            QuantumCircuit aI;
            std::string paulistr = pauli_idx_str(to_base4(I), ref.size());
            if(paulistr.length() != ref.size()){
                throw std::invalid_argument("paulistr.length() != ref.size()");
            }

            for(int k=0; k<ref.size(); k++){
                if(paulistr.substr(k,1) != "0"){
                    aI.add_gate(make_gate(paulis[paulistr.substr(k, 1)], k, k));
                }
            }
            AI.add_term(1.0, aI);
            add_term(1.0, AI);
        }
    } else if(pool_type == "cqoy") {
        std::map<std::string, std::string> paulis = {{"0","I"}, {"1","X"}, {"2","Y"}, {"3","Z"}};
        int nterms = static_cast<int>(std::pow(4, ref.size()));

        for(int I=0; I<nterms; I++){
            QuantumOperator AI;
            QuantumCircuit aI;
            std::string paulistr = pauli_idx_str(to_base4(I), ref.size());
            if(paulistr.length() != ref.size()){
                throw std::invalid_argument("paulistr.length() != ref.size()");
            }
            int nygates = 0;
            for(int k=0; k<ref.size(); k++){
                if(paulistr.substr(k,1) == "2"){
                    nygates++;
                }
                if(paulistr.substr(k,1) != "0"){
                    aI.add_gate(make_gate(paulis[paulistr.substr(k, 1)], k, k));
                }
            }
            if(nygates % 2 != 0){
                AI.add_term(1.0, aI);
                add_term(1.0, AI);
            }
        }
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

std::string QuantumOpPool::to_base4(int I){
    std::string convert_str = "0123456789";
    if (I < 4){
        return convert_str.substr(I, 1);
    } else {
        return to_base4(std::floor(I/4)) + convert_str.substr(I%4, 1);
    }
}

std::string QuantumOpPool::pauli_idx_str(std::string I_str, int nqb){
    std::string res;
    for(int i=0; i<nqb-I_str.length(); i++){
        res.append("0");
    }
    return res + I_str;
}
