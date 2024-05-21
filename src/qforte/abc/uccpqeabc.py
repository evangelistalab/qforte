"""
UCC-PQE base classes
====================================
The abstract base classes inheritied by any projective quantum eigensolver (PQE)
variant that utilizes a unitary coupled cluster (UCC) type ansatz.
"""

import qforte as qf
from abc import abstractmethod
from qforte.abc.pqeabc import PQE
from qforte.abc.ansatz import UCC

from qforte.experiment import *
from qforte.utils.transforms import *
from qforte.utils.state_prep import ref_to_basis_idx
from qforte.utils.trotterization import trotterize

import numpy as np


class UCCPQE(UCC, PQE):
    """The abstract base class inheritied by any algorithm that seeks to find
    eigenstates by minimization of the residual condition

    .. math::
        r_\\mu = \\langle \\Phi_\\mu | \\hat{U}^\\dagger(\\mathbf{t}) \\hat{H} \\hat{U}(\\mathbf{t}) | \\Phi_0 \\rangle \\rightarrow 0

    using a disentagled UCC type ansatz

    .. math::
        \\hat{U}(\\mathbf{t}) = \\prod_\\mu e^{t_\\mu (\\hat{\\tau}_\\mu - \\hat{\\tau}_\\mu^\\dagger)},

    were :math:`\\hat{\\tau}_\\mu` is a Fermionic excitation operator and
    :math:`t_\\mu` is a cluster amplitude.

    Attributes
    ----------

    _noise_factor : float
        The standard deviation of a normal distribution from which noisy residual
        values may be sampled from. Is zero by default such that all residuals
        are exact.

    _orb_e : list floats
        The Hartree-Fock orbital energies

    """

    def verify_required_UCCPQE_attributes(self):
        if not hasattr(self, "_pool_type"):
            raise NotImplementedError(
                "Concrete UCCPQE class must define self._pool_type attribute."
            )

        if not hasattr(self, "_pool_obj"):
            raise NotImplementedError(
                "Concrete UCCPQE class must define self._pool_obj attribute."
            )

    # TODO: consider moving functions from uccnpqe or spqe into this class to
    #      to prevent duplication of code

    def report_iteration(self, x):
        # Printing function for residual minimization. This function is passed to scipy minimize
        # as a callback

        self._k_counter += 1

        if self._k_counter == 1:
            print(
                "\n    k iteration         Energy               dE           Nrvec ev      Nrm ev*         ||r||"
            )
            print(
                "--------------------------------------------------------------------------------------------------"
            )
            if self._print_summary_file:
                f = open("summary.dat", "w+", buffering=1)
                f.write(
                    "\n#    k iteration         Energy               dE           Nrvec ev      Nrm ev*         ||r||"
                )
                f.write(
                    "\n#--------------------------------------------------------------------------------------------------"
                )
                f.close()

        self._curr_energy = self.energy_feval(x)
        dE = self._curr_energy - self._prev_energy
        print(
            f"     {self._k_counter:7}        {self._curr_energy:+12.10f}      {dE:+12.10f}      {self._res_vec_evals:4}        {self._res_m_evals:6}       {self._res_vec_norm:+12.10f}"
        )

        if self._print_summary_file:
            f = open("summary.dat", "a", buffering=1)
            f.write(
                f"\n       {self._k_counter:7}        {self._curr_energy:+12.12f}      {dE:+12.12f}      {self._res_vec_evals:4}        {self._res_m_evals:6}       {self._res_vec_norm:+12.12f}"
            )
            f.close()

        self._prev_energy = self._curr_energy

    def get_sum_residual_square(self, tamps):
        # This function is passed to scipy minimize for residual minimization
        residual_vector = self.get_residual_vector(tamps)
        sum_residual_vector_square = np.sum(np.square(np.abs(residual_vector)))
        assert sum_residual_vector_square.imag < 1.0e-14
        return np.real(sum_residual_vector_square)

    def solve(self):
        if self._optimizer.lower() == "jacobi":
            self.jacobi_solver()
        elif self._optimizer.lower() in [
            "nelder-mead",
            "powell",
            "bfgs",
            "l-bfgs-b",
            "cg",
            "slsqp",
        ]:
            self.scipy_solver(self.get_sum_residual_square)
        else:
            raise NotImplementedError(
                "Currently only Jacobi, Nelder-Mead, Powell, BFGS, L-BFGS-B, CG, and SLSQP solvers are implemented"
            )
