#ifndef _qubit_op_pool_h_
#define _qubit_op_pool_h_

#include <complex>
#include <string>
#include <vector>

class Gate;
class SQOperator;
class QubitOperator;

class QubitOpPool {
    /* A QubitOpPool is a set of QubitOperators, equipped with utility functions for
     * common operations on these objects.
     */
  public:
    /// default constructor: creates an empty second quantized operator pool
    QubitOpPool() {}

    /// set all the terms of the QubitOperator from a vector of QubitOperators
    void set_terms(std::vector<std::pair<std::complex<double>, QubitOperator>>& new_terms);

    /// Add a QubitOperator, and optionally a description
    void add_term(std::complex<double> coeff, const QubitOperator& q_op, const std::string& str = "");

    /// sets the operator pool coefficeints
    void set_coeffs(const std::vector<std::complex<double>>& new_coeffs);

    /// sets the operator pool coefficeints
    void set_op_coeffs(const std::vector<std::complex<double>>& new_coeffs);

    /// return a vector of terms and their coefficients
    const std::vector<std::pair<std::complex<double>, QubitOperator>>& terms() const;

    /// return a vector of QubitOperators multiplied by thier coefficients
    const std::vector<std::pair< std::complex<double>, QubitOperator>>& operator_terms() const;

    /// Convert a pool into an operator
    QubitOperator get_qubit_operator(const std::string& order_type, bool combine_like_terms=true);

    /// join an operator to all terms from the right as (i.e. term -> term*Op)
    /// without simplifying
    void join_op_from_right_lazy(const QubitOperator& q_op);

    /// join an operator to all terms from the right as (i.e. term -> term*Op)
    void join_op_from_right(const QubitOperator& q_op);

    /// join an operator to all terms from the left (i.e. term -> Op*term)
    void join_op_from_left(const QubitOperator& q_op);

    /// join an operator to all terms to form the commutator (i.e. term -> [term, Op])
    void join_as_commutator(const QubitOperator& q_op);

    /// square the current operator pool,
    void square(bool upper_triangle_only);

    /// builds the quantum operator pool, will be used in qite
    void fill_pool(std::string pool_type, const std::vector<int>& ref);

    /// return a vector of strings representing this quantum operator pool
    std::string str() const;

    const std::string& get_description(size_t i) { return descriptions_[i] ;} ;

  private:
    /// the list of sq operators in the pool
    std::vector<std::pair<std::complex<double>, QubitOperator>> terms_;

    /// descriptions of each operator in the pool. optional
    std::vector<std::string> descriptions_;

    /// returns a string representing I in base 4
    std::string to_base4(int I);

    /// fixes the number of preceding zeros in I_str based on nqb
    std::string pauli_idx_str(std::string I_str, int nqb);
};

#endif // _qubit_op_pool_h_
