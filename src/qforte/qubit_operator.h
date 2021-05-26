#ifndef _qubit_operator_h_
#define _qubit_operator_h_

#include <complex>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>

class QubitOperator {
    /* A QubitOperator is a linear combination (over C) of quantum circuits,
     * and therefore a linear combination of products of quantum gates.
     */
  public:
    /// default constructor: creates an empty quantum operator
    QubitOperator() {}

    /// build from a string of open fermion qubit operators
    void build_from_openferm_str(std::string op) {}

    /// build from an openfermion qubit operator directly
    /// might make this a python function?
    void build_from_openferm(std::string op) {}

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
    void simplify(bool combine_like_terms=true);

    /// Multiply this operator (on the right) by rqo.
    /// pre_simplify simplifies this operator before the multiplication.
    /// post_simplify simplifies this operator after the multiplication, as opposed to just canoncalizing.
    void operator_product(const QubitOperator& rqo, bool pre_simplify = true, bool post_simplify = true);

    /// check if this operator is equivalent to another operator qo
    /// mostly used for testing
    bool check_op_equivalence(QubitOperator qo, bool reorder);

    /// return a string representing this quantum operator
    std::string str() const;

    size_t num_qubits() const;

  private:
    /// the linear combination of circuits
    std::vector<std::pair<std::complex<double>, Circuit>> terms_;
};

#endif // _qubit_operator_h_
