"""
vqeabc.py
====================================
The abstract base class inherited by subclasses that
execute general variational quantum eigensolvers.
"""
from abc import abstractmethod
from qforte.abc.algorithm import Algorithm

class VQE(Algorithm):
    """
    Attributes
    ----------
    _ompimizer : string
        The type of optimizer to use for the classical portion of VQE. Suggested
        algorithms are 'BFGS' or 'Nelder-Mead' although there are many options
        (see SciPy.optimize.minimize documentation).

    _converged : bool
        Whether or not the classical optimzation has converged

    _final_result : object
        The result object returned by the scipy optimizer at the end of the
        optimization.

    _opt_maxiter : int
        The maximum number of iterations for the classical optimizer.

    _opt_thresh : float
        The numerical convergence threshold for the specified classical
        optimization algorithm. Is usually the norm of the gradient, but
        is algorithm dependant, see scipy.minimize.optimize for detials.

    Methods
    -------
    build_Uvqc()
        Returns the QuantumCircuit object corresponding to the variational
        quantum circuit unitary (Uprep) used to prepare the VQE state.

    measure_gradient()
        Returns the energy gradient aray pertaining to the variational
        paramaters used in the preparation circuit Uvqc.

    measure_energy()
        Returns the energy expectation value of the state prepared with Uvqc.

    energy_feval()
        The cost function called by the optimizer to be minimized.

    gradient_ary_feval()
        The gradeint function called by the optimizer.

    solve()
        Exectues the overall optimization, encompassing iterative evaluation of
        the energy and gradients, and update of the parameters.

    """

    @abstractmethod
    def build_Uvqc(self):
        pass

    @abstractmethod
    def measure_gradient(self):
        pass

    @abstractmethod
    def measure_energy(self):
        pass

    @abstractmethod
    def energy_feval(self):
        pass

    @abstractmethod
    def gradient_ary_feval(self):
        pass

    @abstractmethod
    def solve(self):
        pass

    def verify_required_VQE_attributes(self):
        if self._optimizer is None:
            raise NotImplementedError('Concrete VQE class must define self._optimizer attribute.')

        if self._converged is None:
            raise NotImplementedError('Concrete VQE class must define self._converged attribute.')

        if self._final_result is None:
            raise NotImplementedError('Concrete VQE class must define self._final_result attribute.')

        if self._opt_maxiter is None:
            raise NotImplementedError('Concrete VQE class must define self._opt_maxiter attribute.')

        if self._opt_thresh is None:
            raise NotImplementedError('Concrete VQE class must define self._opt_thresh attribute.')
