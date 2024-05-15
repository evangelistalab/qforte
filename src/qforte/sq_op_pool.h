#ifndef _sq_op_pool_h_
#define _sq_op_pool_h_

#include <complex>
#include <string>
#include <vector>

class SQOperator;
class QubitOperator;
class QubitOpPool;

// Represents an arbitrary linear combination of second quantized operators.
// May also represent an array of second quantized operators by ignoring
// the coefficients.
class SQOpPool {
  public:
    /// default constructor: creates an empty second quantized operator pool
    SQOpPool() {}

    /// add one set of annihilators and/or creators to the second quantized operator pool
    void add_term(std::complex<double> coeff, const SQOperator& sq_op);

    /// sets the operator pool coefficients
    void set_coeffs(const std::vector<std::complex<double>>& new_coeffs);

    /// return a vector of terms and their coeficients
    const std::vector<std::pair<std::complex<double>, SQOperator>>& terms() const;

    /// set the total number of occupied and virtual spatial orbitals from a reference, from the
    /// number
    ///     of occupied spin orbitals of each point group symmetry
    void set_orb_spaces(const std::vector<int>& ref,
                        const std::vector<size_t>& orb_irreps_to_int = {});

    /// returns a QubitOpPool object with one term for each term in terms_
    QubitOpPool get_qubit_op_pool();

    /// returns a single QubitOperator of the JW transformed sq ops
    QubitOperator get_qubit_operator(const std::string& order_type, bool combine_like_terms = true,
                                     bool qubit_excitations = false);

    /// builds the sq operator pool
    void fill_pool(std::string pool_type);

    /// return a vector of string representing this sq operator pool
    std::string str() const;

  private:
    /// the integer representing the refrence determinant
    uint64_t ref_int_;

    /// the number of spinorbitals
    int n_spinorb_;

    /// the number of occupied alpha spinorbitals
    int n_occ_alpha_;

    /// the number of occupied beta spinorbitals
    int n_occ_beta_;

    /// the number of virtual alpha spinorbitals
    int n_vir_alpha_;

    /// the number of virtual beta spinorbitals
    int n_vir_beta_;

    /// the list of integers representing the irreps of the orbitals
    std::vector<size_t> orb_irreps_to_int_;

    /// the list of sq operators in the pool
    std::vector<std::pair<std::complex<double>, SQOperator>> terms_;
};

#endif // _sq_op_pool_h_
