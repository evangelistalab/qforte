#ifndef _sq_operator_h_
#define _sq_operator_h_

#include <complex>
#include <string>
#include <vector>
#include <numeric>
#include <map>

class QuantumGate;

class QuantumOperator;

class SQOperator {
    /* A SQOperator is a linear combination (over C) of vaccuum-normal, particle-conserving
     * products of fermionic second quantized operators.
     */
  public:
    /// default constructor: creates an empty second quantized operator
    SQOperator() {}

    /// add one product of annihilators and/or creators to the second quantized operator
    void add_term(std::complex<double> coeff, const std::vector<size_t>& cre_ops, const std::vector<size_t>& ann_ops);

    /// add an second quantized operator to the second quantized operator
    void add_op(const SQOperator& sqo);

    /// sets the operator coefficients
    void set_coeffs(const std::vector<std::complex<double>>& new_coeffs);

    /// multiplies the sq operator coefficients by multiplier
    void mult_coeffs(const std::complex<double>& multiplier);

    /// return a vector of terms and their coefficients
    const std::vector<std::tuple< std::complex<double>, std::vector<size_t>, std::vector<size_t>>>& terms() const;

    /// Put a single term into "canonical" form. Canonical form orders orbital indices
    /// descending.
    void canonical_order_single_term(std::tuple< std::complex<double>, std::vector<size_t>, std::vector<size_t>>& term );

    /// Canonicalize each term. The order of the terms is unaffected.
    void canonical_order();

    /// Combine like terms in terms_. As a side-effect, canonicalizes the order.
    void simplify();

    /// Return the QuantumOperator object corresponding the the Jordan-Wigner
    /// transform of this sq operator. Calls simplify as a side-effect.
    QuantumOperator jw_transform();

    /// return a vector of string representing this quantum operator
    std::string str() const;

  private:
    /// The linear combination of second quantized operators. Stored in pairs of
    /// coefficients, and then a vector of N created indices, followed by N annihilated indices.
    /// Orbital indices start at zero.
    std::vector<std::tuple< std::complex<double>, std::vector<size_t>, std::vector<size_t>>> terms_;

    /// Calculate the parity of permutation p
    bool permutive_sign_change(std::vector<int> p) const;

    int canonicalize_helper(std::vector<size_t>& op_list) const;

    /// If operators is a vector of orbital indices, add the corresponding creator
    /// or annihilation qubit operators to holder.
    void jw_helper(QuantumOperator& holder, const std::vector<size_t>& operators, bool creator) const;
};

#endif // _sq_operator_h_
