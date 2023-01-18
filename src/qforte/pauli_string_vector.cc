#include "helpers.h"
#include "pauli_string_vector.h"

std::vector<std::string> PauliStringVector::str() const {
    std::vector<std::string> paulis;
    for (const PauliString& ps: PauliVector_) {
        paulis.push_back(ps.str());
        }
    return paulis;
    }

std::vector<PauliString> PauliStringVector::get_vec() const {return PauliVector_;}

void PauliStringVector::order_terms(){
    std::sort(PauliVector_.begin(), PauliVector_.end());
}

size_t PauliStringVector::num_qubits() const{
    size_t max = 0;
    for (const PauliString& ps: PauliVector_) {
        uint64_t i = 1;
        uint64_t bits = (ps.X() | ps.Z()).get_bits();
        //bits |= bits >> 1;
        //bits |= bits >> 2;
        //bits |= bits >> 4;
        //bits |= bits >> 8;
        //bits |= bits >> 16;
        //bits |= bits >> 32;
        //bits ^= bits >> 1;
        while (bits >> 1 != 0) {
            i++;
            bits >>= 1;
        }
        max = std::max(max, i);
    }
    return max;
}


void PauliStringVector::simplify(){
    order_terms();
    std::vector<PauliString> vec;
    std::complex<double> temp_coeff = PauliVector_[0].coeff();
    for (unsigned int i = 0; i < PauliVector_.size() - 1; i++){
        if (PauliVector_[i].X() == PauliVector_[i+1].X() and PauliVector_[i].Z() == PauliVector_[i+1].Z()){
            temp_coeff += PauliVector_[i+1].coeff();
            if (i == PauliVector_.size() - 2 and std::abs(temp_coeff) > 1.0e-12){
                PauliString unique(PauliVector_[i].X(), PauliVector_[i].Z(), temp_coeff);
                vec.push_back(unique);
            }
        }
        else {
            if (std::abs(temp_coeff) > 1.0e-12) {
                PauliString unique(PauliVector_[i].X(), PauliVector_[i].Z(), temp_coeff);
                vec.push_back(unique);
            }
            temp_coeff = PauliVector_[i+1].coeff();
            if (i == PauliVector_.size() - 2 and std::abs(temp_coeff) > 1.0e-12){
                vec.push_back(PauliVector_[i+1]);
            }
        }
    }
    PauliVector_ = std::move(vec);
}

void PauliStringVector::add_PauliString(const PauliString& ps){
    PauliVector_.push_back(ps);
}

void PauliStringVector::add_PauliStringVector(const PauliStringVector& psv){
    std::vector<PauliString> vec(psv.get_vec());
    PauliVector_.insert(PauliVector_.end(), vec.begin(), vec.end());
    //PauliVector_.insert(PauliVector_.end(), psv.get_vec().begin(), psv.get_vec().end());
}

QubitOperator PauliStringVector::get_QubitOperator(){
    QubitOperator q_op;
    for(PauliString& ps : PauliVector_){
        Circuit temp_circ;
        uint64_t Y_gates = (ps.X() & ps.Z()).get_bits();
        uint64_t X_gates = ps.X().get_bits() ^ Y_gates;
        uint64_t Z_gates = ps.Z().get_bits() ^ Y_gates;
        for (int i = 0; i < 64; i++){
            if (X_gates & 1ULL << i){temp_circ.add_gate(make_gate("X", i, i));}
            else {
                if (Y_gates & 1ULL << i){temp_circ.add_gate(make_gate("Y", i, i));}
                else {
                    if (Z_gates & 1ULL << i){temp_circ.add_gate(make_gate("Z", i, i));}
                }
            }
        }
        q_op.add_term(ps.coeff(), temp_circ);
    }
    return q_op;
}

//void PauliStringVector::right_multiply(const PauliStringVector& rhs){
//    PauliStringVector temp;
//    for(PauliString& ps1 : PauliVector_){
//        for(PauliString& ps2 : rhs.get_vec()){
//            temp.add_PauliString(multiply(ps2,ps1));
//        }
//    }
//
//    PauliVector_ = std::move(temp.get_vec());
//    simplify();
//}

PauliStringVector multiply(const PauliStringVector& lhs, const std::complex<double> rhs){
    std::vector<PauliString> vec;
    for(PauliString& ps : lhs.get_vec()){
        PauliString product(ps.X(), ps.Z(), ps.coeff() * rhs);
        vec.push_back(product);
    }
    PauliStringVector PauliVector(vec);
    return PauliVector;
}

PauliStringVector multiply(const PauliStringVector& lhs, const PauliString& rhs){
    std::vector<PauliString> vec;
    for(PauliString& ps : lhs.get_vec()){
        PauliString product = multiply(rhs, ps);
        vec.push_back(product);
    }
    PauliStringVector PauliVector(vec);
    return PauliVector;
}

PauliStringVector multiply(const PauliString& lhs, const PauliStringVector& rhs){
    std::vector<PauliString> vec;
    for(PauliString& ps : rhs.get_vec()){
        PauliString product = multiply(ps, lhs);
        vec.push_back(product);
    }
    PauliStringVector PauliVector(vec);
    return PauliVector;
}

PauliStringVector multiply(const PauliStringVector& lhs, const PauliStringVector& rhs){
    std::vector<PauliString> vec;
    for(PauliString& ps1 : lhs.get_vec()){
        for(PauliString& ps2 : rhs.get_vec()){
            PauliString product = multiply(ps1, ps2);
            vec.push_back(product);
        }
    }
    PauliStringVector PauliVector(vec);
    return PauliVector;
}

PauliStringVector add(const PauliStringVector& lhs, const PauliString& rhs){
    std::vector<PauliString> vec(lhs.get_vec());
    vec.push_back(rhs);
    PauliStringVector PauliVector(vec);
    return PauliVector;
}

PauliStringVector add(const PauliStringVector& lhs, const PauliStringVector& rhs){
     std::vector<PauliString> vec1(lhs.get_vec());
     std::vector<PauliString> vec2(rhs.get_vec());
     vec1.insert(vec1.end(), vec2.begin(), vec2.end());
     PauliStringVector PauliVector(vec1);
     return PauliVector;
}

PauliStringVector subtract(const PauliStringVector& lhs, const PauliString& rhs){
    std::vector<PauliString> vec(lhs.get_vec());
    PauliString rhs_minus(rhs.X(), rhs.Z(), -rhs.coeff());
    vec.push_back(rhs_minus);
    PauliStringVector PauliVector(vec);
    return PauliVector;
}

PauliStringVector subtract(const PauliString& lhs, const PauliStringVector& rhs){
    std::vector<PauliString> vec{lhs};
    for(const PauliString& ps: rhs.get_vec()){
        PauliString ps_minus(ps.X(), ps.Z(), -ps.coeff());
        vec.push_back(ps_minus);
    }
    PauliStringVector PauliVector(vec);
    return PauliVector;
}

PauliStringVector subtract(const PauliStringVector& lhs, const PauliStringVector& rhs){
    std::vector<PauliString> vec(lhs.get_vec());
    for(const PauliString& ps: rhs.get_vec()){
        PauliString ps_minus(ps.X(), ps.Z(), -ps.coeff());
        vec.push_back(ps_minus);
    }
    PauliStringVector PauliVector(vec);
    return PauliVector;
}
