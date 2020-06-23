#ifndef _sq_op_pool_h_
#define _sq_op_pool_h_

#include <complex>
#include <string>
#include <vector>

class SQOperator;

class QuantumOperator;

class SQOpPool {
  public:
    /// default constructor: creates an empty second quantized operator pool
    SQOpPool() {}

    /// add one set of anihilators and/or creators to the second quantized operator pool
    void add_term(std::complex<double> coeff, const SQOperator& sq_op );

    /// sets the operator pool coefficeints
    void set_coeffs(const std::vector<std::complex<double>>& new_coeffs);

    /// return a vector of terms and thier coeficients
    const std::vector<std::pair< std::complex<double>, SQOperator>>& terms() const;

    /// set the orbtial occupations from a reference
    void set_orb_spaces(const std::vector<int>& ref);

    /// returns the JW transformed vector of operators
    std::vector<QuantumOperator> get_quantum_operators();

    /// return a singe QuantumOperator of tge JW transformed ops
    QuantumOperator get_quantum_operator();

    /// builds the sq operator pool
    void fill_pool(std::string pool_type);

    /// return a vector of string representing this sq operator pool
    std::string str() const;

  private:
    /// the number of occupied spatial orbitals
    int nocc_;

    /// the number of virtual spatial orbitals
    int nvir_;

    /// the list of sq operators in the pool
    std::vector<std::pair<std::complex<double>, SQOperator>> terms_;

};

#endif // _sq_op_pool_h_
