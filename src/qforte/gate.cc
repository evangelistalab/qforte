#include "fmt/format.h"

#include "gate.h"
#include "qubit_basis.h"
#include "sparse_tensor.h"

const std::vector<std::pair<size_t, size_t>> Gate::two_qubits_basis_{
    {0, 0}, {0, 1}, {1, 0}, {1, 1}};
const std::vector<size_t> Gate::index1{0, 1};
const std::vector<size_t> Gate::index2{0, 1, 2, 3};

Gate::Gate(const std::string& label, size_t target, size_t control, std::complex<double> gate[4][4],
           std::optional<std::pair<double, bool>> parameter)
    : label_(label), target_(target), control_(control), parameter_(parameter) {
    for (const auto& i : index2) {
        for (const auto& j : index2) {
            gate_[i][j] = gate[i][j];
        }
    }
}

size_t Gate::target() const { return target_; }

size_t Gate::control() const { return control_; }

const complex_4_4_mat& Gate::matrix() const { return gate_; }

const SparseMatrix Gate::sparse_matrix(size_t nqubit) const {
    size_t nbasis = std::pow(2, nqubit);
    if (target_ != control_) {
        throw std::runtime_error("Gate must be a Pauli to convert to matrix!");
    } else if (target_ >= nqubit) {
        throw std::runtime_error("Target index is too large for specified nqbits!");
    }

    SparseMatrix Spmat = SparseMatrix();

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            auto op_i_j = gate_[i][j];
            if (std::abs(op_i_j) > 1.0e-16) {
                for (size_t I = 0; I < nbasis; I++) {
                    QubitBasis basis_I = QubitBasis(I);
                    if (basis_I.get_bit(target_) == j) {
                        QubitBasis basis_J = basis_I;
                        basis_J.set_bit(target_, i);
                        Spmat.set_element(basis_J.add(), basis_I.add(), op_i_j);
                    }
                }
            }
        }
    }
    return Spmat;
}

std::string Gate::gate_id() const { return label_; }

bool Gate::has_parameter() const { return parameter_.has_value(); }

std::optional<double> Gate::parameter() const {
    return has_parameter() ? std::make_optional(parameter_.value().first) : std::nullopt;
}

bool Gate::minus_parameter_on_adjoint() const {
    return has_parameter() ? parameter_.value().second : false;
}

std::string Gate::str() const {
    if (target_ == control_) {
        return fmt::format("{}{}", label_, target_);
    }
    return fmt::format("{}{}_{}", label_, target_, control_);
}

std::string Gate::repr() const {
    std::string s =
        fmt::format("{} gate, target qubit:{}, contol qubit:{}\n", label_, target_, control_);
    const std::vector<size_t>& index = (nqubits() == 1 ? index1 : index2);
    for (const auto& i : index) {
        for (const auto& j : index) {
            s += fmt::format("  {:+f} {:+f} i", std::real(gate_[i][j]), std::imag(gate_[i][j]));
        }
        s += '\n';
    }
    return s;
}

size_t Gate::nqubits() const { return (target_ == control_ ? 1 : 2); }

Gate Gate::adjoint() const {
    std::complex<double> adj_gate[4][4];
    bool self_adjoint = true;
    for (const auto& i : index2) {
        for (const auto& j : index2) {
            adj_gate[j][i] = std::conj(gate_[i][j]);
            if (std::norm(adj_gate[j][i] - gate_[i][j]) > 1.0e-12) {
                self_adjoint = false;
            }
        }
    }
    auto parameter_info = has_parameter()
                              ? std::make_optional(std::make_pair(
                                    parameter_.value().first *
                                        static_cast<double>(1 - 2 * minus_parameter_on_adjoint()),
                                    minus_parameter_on_adjoint()))
                              : std::nullopt;
    if (not self_adjoint) {
        // check if label_ is of the form adj(x) and if it is then return Gate(x)
        if (label_.size() > 4 and label_.substr(0, 4) == "adj(" and label_.back() == ')') {
            return Gate(label_.substr(4, label_.size() - 5), target_, control_, adj_gate,
                        parameter_info);
        }
        return Gate("adj(" + label_ + ")", target_, control_, adj_gate, parameter_info);
    }

    return Gate(label_, target_, control_, adj_gate, parameter_info);
}

bool Gate::operator==(const Gate& rhs) const {
    if (target_ != rhs.target_) {
        return false;
    }
    if (control_ != rhs.control_) {
        return false;
    }
    for (const auto& i : index2) {
        for (const auto& j : index2) {
            if (std::norm(gate_[i][j] - rhs.gate_[i][j]) > 1.0e-12) {
                return false;
            }
        }
    }
    return true;
}

const std::vector<std::pair<size_t, size_t>>& Gate::two_qubits_basis() { return two_qubits_basis_; }
