#ifndef _quantum_op_pool_h_
#define _quantum_op_pool_h_

#include <complex>
#include <string>
#include <vector>

class Gate;
class SQOperator;
class QuantumOperator;

class QuantumOpPool {
    /* A QuantumOpPool is a set of QuantumOperators, equipped with utility functions for
     * common operations on these objects.
     */
  public:
    /// default constructor: creates an empty second quantized operator pool
    QuantumOpPool() {}

    /// set all the terms of the QuantumOperator from a vector of QuantumOperators
    void set_terms(std::vector<std::pair<std::complex<double>, QuantumOperator>>& new_terms);

    /// add one set of anihilators and/or creators to the second quantized operator pool
    void add_term(std::complex<double> coeff, const QuantumOperator& q_op );

    /// sets the operator pool coefficeints
    void set_coeffs(const std::vector<std::complex<double>>& new_coeffs);

    /// sets the operator pool coefficeints
    void set_op_coeffs(const std::vector<std::complex<double>>& new_coeffs);

    /// return a vector of terms and their coefficients
    const std::vector<std::pair<std::complex<double>, QuantumOperator>>& terms() const;

    /// return a vector of QuantumOperators multiplied by thier coefficients
    const std::vector<std::pair< std::complex<double>, QuantumOperator>>& operator_terms() const;

    /// join an operator to all terms from the right as (i.e. term -> term*Op)
    /// without simplifying
    void join_op_from_right_lazy(const QuantumOperator& q_op);

    /// join an operator to all terms from the right as (i.e. term -> term*Op)
    void join_op_from_right(const QuantumOperator& q_op);

    /// join an operator to all terms from the left (i.e. term -> Op*term)
    void join_op_from_left(const QuantumOperator& q_op);

    /// join an operator to all terms to form the commutator (i.e. term -> [term, Op])
    void join_as_commutator(const QuantumOperator& q_op);

    /// square the current operator pool,
    void square(bool upper_triangle_only);

    /// builds the quantum operator pool, will be used in qite
    void fill_pool(std::string pool_type, const std::vector<int>& ref);

    /// return a vector of strings representing this quantum operator pool
    std::string str() const;

  private:
    /// the list of sq operators in the pool
    std::vector<std::pair<std::complex<double>, QuantumOperator>> terms_;

    /// returns a string representing I in base 4
    std::string to_base4(int I);

    /// fixes the number of preceding zeros in I_str based on nqb
    std::string pauli_idx_str(std::string I_str, int nqb);

};

#endif // _quantum_op_pool_h_
