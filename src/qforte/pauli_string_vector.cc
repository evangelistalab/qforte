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

//unsigned int PauliStringVector::size() const {return PauliVector_.size();}
//
//unsigned int PauliStringVector::begin() const {return PauliVector_.begin();}
//unsigned int PauliStringVector::end() const {return PauliVector_.end();}

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
