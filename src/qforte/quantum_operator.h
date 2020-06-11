#ifndef _quantum_operator_h_
#define _quantum_operator_h_

#include <complex>
#include <string>
#include <vector>
// #include <map>
// #include <unordered_map>

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

    /// return a vector of terms and thier coeficients
    const std::vector<std::pair<std::complex<double>, QuantumCircuit>>& terms() const;

    /// order each product of ac operators in a standardized fashion
    void canonical_order();

    /// adds term to uniqe_trms if not already contained in uniqe_trms, otherwise
    /// adds coeficent of term to coeficent of identical term in uniqe_trms
    void add_unique_term(
        std::vector<std::pair<std::complex<double>, QuantumCircuit>>& uniqe_trms,
        const std::pair<std::complex<double>, QuantumCircuit>& term
    );

    /// simplify the operator (i.e. combine like terms)
    void simplify();

    /// return a vector of string representing this quantum operator
    std::string str() const;

  private:
    /// the list of circuits
    std::vector<std::pair<std::complex<double>, QuantumCircuit>> terms_;
};

#endif // _quantum_operator_h_
