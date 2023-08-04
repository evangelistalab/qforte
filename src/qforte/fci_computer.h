#ifndef _fci_computer_h_
#define _fci_computer_h_

#include <string>
#include <vector>

#include "qforte-def.h" 
#include "tensor.h" 
#include "fci_graph.h" 


class Gate;
class QubitBasis;
class SQOperator;
class TensorOperator;
class Tensor;
class FCIGraph;

class FCIComputer {
  public:
    /// default constructor: create a 'FCI' quantum computer 
    /// the computer represends a restricted hilbert space for 'chemistry'
    /// nel: the number of electrons
    /// sz: the z componant spin
    /// norb: the number of spatial orbitals 
    /// Implementation will be reminicient of modenrn determinant CI codes
    /// Implementation also borrows HEAVILY from the fermionic quantum emulator wfn class
    /// see (https://quantumai.google/openfermion/fqe) and related article
    FCIComputer(int nel, int sz, int norb);

    /// apply a SQOperator to the current state.
    /// (this operation is generally not a physical quantum computing operation).
    /// Only works if the sqo conservs particle number and multiplicity.
    /// TODO(Tyler?): implement this...
    // void apply_sq_operator(const QubitOperator& qo);

    /// apply a TensorOperator to the current state 
    void apply_tensor_operator(const TensorOperator& top);

    /// apply a 1-body TensorOperator to the current state 
    void apply_tensor_spin_1bdy(const TensorOperator& top);

    void apply_tensor_spin_1bdy(const Tensor& h1e, size_t norb);

    void lm_apply_array1(
      // const double complex *coeff, don't need
      // double complex *out,
      const Tensor& out,
      // const int *dexc,
      const std::vector<int> dexc,
      const int astates,
      const int bstates,
      const int ndexc,
      // const double complex *h1e,
      const Tensor& h1e,
      const int norbs,
      const bool is_alpha);
      // const struct blasfunctions * blasfunc);

    void lm_apply_array1_new(
      Tensor& out,
      const std::vector<int>& dexc,
      const int astates,
      const int bstates,
      const int ndexc,
      const Tensor& h1e,
      const int norbs,
      const bool is_alpha);

    /// apply a 1-body and 2-body TensorOperator to the current state 
    void apply_tensor_spin_12_body(const TensorOperator& top);

    /// apply a second quantized operator, must be number and spin conserving.
    void apply_sqop(const SQOperator& sqop);

    /// apply a constant to the FCI quantum computer.
    void scale(const std::complex<double> a);

    /// measure the expecation value (exactly) of the FCI quantum computer 
    /// with respect a TensorOperator
    std::vector<double> direct_expectation_value(const TensorOperator& top);

    // NOTE(Nick): may need these Later!
    /// Measure expectation value of all operators in an operator pool
    // std::vector<std::complex<double>> direct_oppl_exp_val(const QubitOpPool& qopl);

    /// measure expectation value for specific operators in an operator pool
    // std::vector<std::complex<double>> direct_idxd_oppl_exp_val(const QubitOpPool& qopl, const std::vector<int>& idxs);

    /// measure expectaion value of all operators in an operator pool, where the
    /// operator coefficents have been multipild by mults
    // std::vector<std::complex<double>> direct_oppl_exp_val_w_mults(
    //     const QubitOpPool& qopl,
    //     const std::vector<std::complex<double>>& mults);

    /// return a string representing the state of the computer
    /// TODO(Nick) Implement (this will be a pain :/)
    std::string str(
      bool print_data,
      bool print_complex
      ) const 
    {
      return C_.str(print_data, print_complex); 
    }

    /// return a tensor of the coeficients
    Tensor get_state() const { return C_; };

    /// return the coefficient corresponding to a alpha-basis / beta-basis 
    std::complex<double> coeff(const QubitBasis& abasis, const QubitBasis& bbasis);

    /// return the number of electrons
    size_t get_nel() const { return nel_; }

    /// return the z-componant spin
    size_t get_sz() const { return sz_; }
    
    /// return the number of spatial orbitals
    size_t none_ops() const { return norb_; }

    /// return the number of two-qubit operations
    /// NOTE(Nick) Maybe try to keep (or some proxy at least)?
    // size_t ntwo_ops() const { return ntwo_ops_; }

    /// set the coefficient tensor directly from another coefficient tensor
    /// checks the shape, throws if incompatable
    void set_state(const Tensor other_state);

    /// set the quantum computer to the state
    /// basis_1 * c_1 + basis_2 * c_2 + ...
    /// where this information is passed as a vectors of pairs
    ///  [(basis_1, c_1), (basis_2, c_2), ...]
    // void set_state(std::vector<std::pair<QubitBasis, double_c>> state);

    /// Sets all coefficeints fo the FCI Computer to Zero
    void zero();

    /// Sets all coefficeints fo the FCI Computer to Zero except the HF Determinant (set to 1).
    void hartree_fock();

    /// get timings
    std::vector<std::pair<std::string, double>> get_timings() { return timings_; }

    /// clear the timings
    void clear_timings() { timings_.clear(); }

  private:
    /// the number of electrons
    size_t nel_;

    /// the number of alpha electrons
    size_t nalfa_el_;

    /// the number of beta electrons
    size_t nbeta_el_;

    /// the number of alpha strings
    size_t nalfa_strs_;

    /// the number of beta strings
    size_t nbeta_strs_;

    /// the z-componant spin
    size_t sz_;

    /// the number of spatial orbitals
    size_t norb_;

    /// the number of alpha basis states 
    size_t nabasis_;

    /// the number of beta basis states 
    size_t nbbasis_;

    /// NOTE(Nick): Unsure if needed, may be repalced by FCIGraph?
    /// the tensor product basis
    // std::vector<QubitBasis> basis_;

    // The name of the FCI Shape;
    const std::string name_ = "FCIComputer State";

    /// The coefficients of the starting state in the tensor product basis
    Tensor C_;

    /// The corresponding FCIGraph for this computer
    FCIGraph graph_;

    /// the coefficients of the ending state in the tensor product basis
    // std::vector<std::complex<double>> new_coeff_;

    /// timings and descriptions accessable in python
    std::vector<std::pair<std::string, double>> timings_;

    /// NOTE(Nick): keeping for future?
    /// the number of one-qubit operations
    // size_t none_ops_ = 0;
    /// the number of two-qubit operations
    // size_t ntwo_ops_ = 0;

    /// the threshold for doing operations with elements of gate matricies
    double compute_threshold_ = 1.0e-16;
};

#endif // _fci_computer_h_
