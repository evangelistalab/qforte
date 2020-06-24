#ifndef _quantum_op_pool_h_
#define _quantum_op_pool_h_

#include <complex>
#include <string>
#include <vector>

class SQOperator;

class QuantumOperator;

class QuantumOpPool {
  public:
    /// default constructor: creates an empty second quantized operator pool
    QuantumOpPool() {}

    /// set all the terms of the QuantumOperator from a vector of QuantumOperators
    void set_terms(std::vector<std::pair<std::complex<double>, QuantumOperator>>& new_terms);

    /// add one set of anihilators and/or creators to the second quantized operator pool
    void add_term(std::complex<double> coeff, const QuantumOperator& q_op );

    /// sets the operator pool coefficeints
    void set_coeffs(const std::vector<std::complex<double>>& new_coeffs);

    /// return a vector of terms and thier coeficients
    const std::vector<std::pair<std::complex<double>, QuantumOperator>>& terms() const;

    /// return a vector of QuantumOperators multiplied by thier coeficients
    const std::vector<std::pair< std::complex<double>, QuantumOperator>>& operator_terms() const;

    /// set the orbtial occupations from a reference
    void set_orb_spaces(const std::vector<int>& ref);

    /// join an operator to all terms from the right as (i.e. term -> term*Op)
    void join_op_from_right(const QuantumOperator& q_op);

    /// join an operator to all terms from the left (i.e. term -> Op*term)
    void join_op_from_left(const QuantumOperator& q_op);

    /// join an operator to all terms to form the comutator (i.e. term -> [term, Op])
    void join_as_comutator(const QuantumOperator& q_op);

    /// builds the quantum operator pool, will be used in qite
    void fill_pool(std::string pool_type);

    /// return a vector of strings representing this quantum operator pool
    std::string str() const;

  private:
    /// the number of occupied spatial orbitals
    int nocc_;

    /// the number of virtual spatial orbitals
    int nvir_;

    /// the list of sq operators in the pool
    std::vector<std::pair<std::complex<double>, QuantumOperator>> terms_;

};

#endif // _quantum_op_pool_h_
