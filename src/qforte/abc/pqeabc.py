"""
PQE base classes
====================================
The abstract base classes inheritied by any projective quantum eigensolver (PQE)
variant.
"""

from abc import abstractmethod
from qforte.abc.algorithm import AnsatzAlgorithm


class PQE(AnsatzAlgorithm):
    """The abstract base class inheritied by any algorithm that seeks to find
    eigenstates by minimization of the residual condition

    .. math::
        r_\mu = \langle \Phi_\mu | \hat{U}^\dagger(\mathbf{\\theta}) \hat{H} \hat{U}(\mathbf{\\theta}) | \Phi_0 \\rangle \\rightarrow 0

    using a general paramaterized ansatz :math:`\hat{U}(\\theta)`.
    """

    @abstractmethod
    def solve(self):
        pass

    def verify_required_PQE_attributes(self):
        if not hasattr(self, "_converged"):
            raise NotImplementedError(
                "Concrete PQE class must define self._converged attribute."
            )

        if not hasattr(self, "_opt_maxiter"):
            raise NotImplementedError(
                "Concrete PQE class must define self._opt_maxiter attribute."
            )

        if not hasattr(self, "_opt_thresh"):
            raise NotImplementedError(
                "Concrete PQE class must define self._opt_thresh attribute."
            )
