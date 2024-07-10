"""
VQE base classes
====================================
The abstract base classes inheritied by any projective variational quantum
eigensolver (VQE) variant.
"""

from abc import abstractmethod
from qforte.abc.algorithm import AnsatzAlgorithm


class VQE(AnsatzAlgorithm):
    """The abstract base class inheritied by any algorithm that seeks to find
    eigenstates by variational minimization of the energy

    .. math::
        E(\\theta) = \langle \Phi_0 | \hat{U}^\dagger(\mathbf{\\theta}) \hat{H} \hat{U}(\mathbf{\\theta}) | \Phi_0 \\rangle

    for a general parameterized unitary :math:`\hat{U}(\\theta)`.

    Attributes
    ----------
    _optimizer : string
        The type of optimizer to use for the classical portion of VQE. Suggested
        algorithms are 'BFGS' or 'Nelder-Mead' although there are many options
        (see SciPy.optimize.minimize documentation).

    _converged : bool
        Whether or not the classical optimzation has converged

    _final_result : object
        The result object returned by the scipy optimizer at the end of the
        optimization.

    """

    @abstractmethod
    def measure_gradient(self):
        """Returns the energy gradient aray pertaining to the variational
        paramaters used in the preparation circuit Uvqc.
        """
        pass

    @abstractmethod
    def gradient_ary_feval(self):
        """Computes the gradients with respect to all operators currently in the
        UCCN-VQE ansatz. Used as the jacobian the minimizer calls.
        """
        pass

    @abstractmethod
    def solve(self):
        """Runs the optimizer to mimimize the energy. Sets certain optimizer
        parameters internally.
        """
        pass

    def verify_required_VQE_attributes(self):
        """Verifies all VQE specific attributes were defined by concrete classes."""
        if self._optimizer is None:
            raise NotImplementedError(
                "Concrete VQE class must define self._optimizer attribute."
            )

        if self._converged is None:
            raise NotImplementedError(
                "Concrete VQE class must define self._converged attribute."
            )

        if self._opt_maxiter is None:
            raise NotImplementedError(
                "Concrete VQE class must define self._opt_maxiter attribute."
            )

        if self._opt_thresh is None:
            raise NotImplementedError(
                "Concrete VQE class must define self._opt_thresh attribute."
            )
