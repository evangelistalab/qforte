#ifndef _quantum_operator_h_
#define _quantum_operator_h_

#include <complex>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>

class QuantumOperator {
  public:
    /// default constructor: creates an empty quantum operator
    QuantumOperator() {}

    /// build from a string of open fermion qubit operators
    void build_from_openferm_str(std::string op) {}

    /// build from an openfermion qubit operator directly
    /// might make this a python function?
    void build_from_openferm(std::string op) {}

    /// add a circuit as a term in the quantum operator
    void add_term(std::complex<double> circ_coeff, const QuantumCircuit& circuit);

    /// add a circuit as a term in the quantum operator
    void add_op(const QuantumOperator& qo);

    /// sets the operator coefficeints
    void set_coeffs(const std::vector<std::complex<double>>& new_coeffs);

    /// multiplies the operator coefficeints by multiplier
    void mult_coeffs(const std::complex<double>& multiplier);

    /// return a vector of terms and thier coeficients
    const std::vector<std::pair<std::complex<double>, QuantumCircuit>>& terms() const;

    /// order the terms by increasing coefficient value
    void order_terms();

    /// order the gates by increasing quibits in each QuantumCircuit in terms_
    /// and contract all pauli operators
    void canonical_order();

    /// simplify the operator using std::unsorted_map (i.e. combine like terms)
    void simplify(bool combine_like_terms=true);

    /// join a new operator to this operator via multiplicaiton
    void join_operator(const QuantumOperator& rqo, bool simplify_lop);

    /// join a new operator to this operator via multiplicaiton without
    /// simplifying the reslut
    void join_operator_lazy(const QuantumOperator& rqo);

    /// check if this operaotr is equivalent to another operator qo
    /// mostly used for testing
    bool check_op_equivalence(QuantumOperator qo, bool reorder);

    /// return a vector of string representing this quantum operator
    std::string str() const;

  private:
    /// the list of circuits
    std::vector<std::pair<std::complex<double>, QuantumCircuit>> terms_;
};

#endif // _quantum_operator_h_
