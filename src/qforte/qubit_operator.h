#ifndef _qubit_operator_h_
#define _qubit_operator_h_

#include <complex>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>

class SparseMatrix;

class QubitOperator {
    /* A QubitOperator is a list of quantum circuits, each of which has a complex scalar.
     * While linear combinations of quantum circuits are QubitOperators, you cannot
     * assume that a QubitOperator is a linear combination: some functions take in a
     * QubitOperator and are well-defined on the list data structure, but not on a linear
     * combination. Example: `trotterize` will give different results if you swap the order
     * of two circuits in the list, even though the linear combination is unchanged.
     */
  public:
    /// default constructor: creates an empty quantum operator
    QubitOperator() {}

    /// add a circuit as a term in the quantum operator
    void add_term(std::complex<double> circ_coeff, const Circuit& circuit);

    /// add the circuits of another quantum operator as a term in the quantum operator
    void add_op(const QubitOperator& qo);

    /// sets the operator coefficients
    void set_coeffs(const std::vector<std::complex<double>>& new_coeffs);

    /// multiplies the operator coefficients by multiplier
    void mult_coeffs(const std::complex<double>& multiplier);

    /// return a vector of terms and their coefficients
    const std::vector<std::pair<std::complex<double>, Circuit>>& terms() const;

    /// order the terms by increasing coefficient value
    void order_terms();

    /// order the gates by increasing qubits in each Circuit in terms_
    /// and contract all pauli operators
    void canonical_order();

    /// Put all operators in the linear combination in canonical form AND THEN
    /// combine like terms.
    void simplify(bool combine_like_terms = true);

    /// Multiply this operator (on the right) by rqo.
    /// pre_simplify simplifies this operator before the multiplication.
    /// post_simplify simplifies this operator after the multiplication, as opposed to just
    /// canoncalizing.
    void operator_product(const QubitOperator& rqo, bool pre_simplify = true,
                          bool post_simplify = true);

    /// check if this operator is equivalent to another operator qo
    /// mostly used for testing
    bool check_op_equivalence(QubitOperator qo, bool reorder);

    /// Returns the lifted sparse matrix representaion of the operator.
    const SparseMatrix sparse_matrix(size_t nqubit) const;

    /// return a string representing this quantum operator
    std::string str() const;

    size_t num_qubits() const;

  private:
    /// the linear combination of circuits
    std::vector<std::pair<std::complex<double>, Circuit>> terms_;
};

#endif // _qubit_operator_h_
