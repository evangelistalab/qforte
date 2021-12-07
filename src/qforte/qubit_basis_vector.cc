#include "helpers.h"
#include "qubit_basis_vector.h"

std::vector<std::string> QubitBasisVector::str() const {
    std::vector<std::string> kets;
    for (const QubitBasis& ket: QBasisVector_) {
        kets.push_back(ket.str(ket.max_qubits()));
        }
    return kets;
    }

std::vector<QubitBasis> QubitBasisVector::get_vec() const {return QBasisVector_;}

//unsigned int QubitBasisVector::size() const {return QBasisVector_.size();}
//
//unsigned int QubitBasisVector::begin() const {return QBasisVector_.begin();}
//unsigned int QubitBasisVector::end() const {return QBasisVector_.end();}

QubitBasisVector multiply(const QubitBasisVector& lhs, const std::complex<double> rhs){
    std::vector<QubitBasis> kets;
    for(QubitBasis& ket : lhs.get_vec()){
        QubitBasis product(ket.get_bits().get_bits(), ket.coeff() * rhs);
        kets.push_back(product);
    }
    QubitBasisVector QBasisVector(kets);
    return QBasisVector;
}

QubitBasisVector add(const QubitBasisVector& lhs, const QubitBasis& rhs){
    std::vector<QubitBasis> kets(lhs.get_vec());
    kets.push_back(rhs);
    QubitBasisVector QBasisVector(kets);
    return QBasisVector;
}

QubitBasisVector add(const QubitBasisVector& lhs, const QubitBasisVector& rhs){
     std::vector<QubitBasis> kets(lhs.get_vec());
     std::vector<QubitBasis> vec(rhs.get_vec());
     kets.insert(kets.end(), vec.begin(), vec.end());
     QubitBasisVector QBasisVector(kets);
     return QBasisVector;
}

QubitBasisVector subtract(const QubitBasisVector& lhs, const QubitBasis& rhs){
    std::vector<QubitBasis> kets(lhs.get_vec());
    QubitBasis rhs_minus(rhs.get_bits().get_bits(), -rhs.coeff());
    kets.push_back(rhs_minus);
    QubitBasisVector QBasisVector(kets);
    return QBasisVector;
}

QubitBasisVector subtract(const QubitBasis& lhs, const QubitBasisVector& rhs){
    std::vector<QubitBasis> kets{lhs};
    for(const QubitBasis& ket: rhs.get_vec()){
        QubitBasis ket_minus(ket.get_bits().get_bits(), -ket.coeff());
        kets.push_back(ket_minus);
    }
    QubitBasisVector QBasisVector(kets);
    return QBasisVector;
}

QubitBasisVector subtract(const QubitBasisVector& lhs, const QubitBasisVector& rhs){
    std::vector<QubitBasis> kets(lhs.get_vec());
    for(const QubitBasis& ket: rhs.get_vec()){
        QubitBasis ket_minus(ket.get_bits().get_bits(), -ket.coeff());
        kets.push_back(ket_minus);
    }
    QubitBasisVector QBasisVector(kets);
    return QBasisVector;
}

QubitBasisVector apply(const QubitBasisVector& lhs, const PauliString& rhs){
    std::vector<QubitBasis> kets;
    for(const QubitBasis& ket: lhs.get_vec()){
        QubitBasis result = apply(rhs, ket);
        kets.push_back(result);
    }
    QubitBasisVector QBasisVector(kets);
    return QBasisVector;
}

QubitBasisVector apply(const QubitBasisVector& lhs, const PauliStringVector& rhs){
    std::vector<QubitBasis> kets;
    for(const PauliString& ps: rhs.get_vec()){
        for(const QubitBasis& ket: lhs.get_vec()){
            QubitBasis result = apply(ps, ket);
            kets.push_back(result);
        }
    }
    QubitBasisVector QBasisVector(kets);
    return QBasisVector;
}
