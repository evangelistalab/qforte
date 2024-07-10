"""
QSD base classes
====================================
The abstract base classes inheritied by any quantum subspace diagonalizaiton (QSD)
variant.
"""

from abc import abstractmethod
from qforte.abc.algorithm import Algorithm
from qforte.maths.eigsolve import canonical_geig_solve

import numpy as np


class QSD(Algorithm):
    """The abstract base class inherited by any algorithm that seeks to find
    eigenstates of the Hamiltonian in a (generally) non-orthogonal basis of
    many-body states :math:`\\{ | \\Psi_n \\rangle \\}`. The basis is generated
    by a corresponding family of unitary operators

    .. math::
        | \\Psi_n \\rangle = \\hat{U}_n | \\Phi_0 \\rangle

    where :math:`| \\Phi \\rangle` is (usually) an unentangled reference
    such as the Hartree-Fock state.

    Quantum subspace diagonalization methods work by constructing an effective
    Hamiltonian :math:`\\mathbf{H}` with matrix elements given by

    .. math::
        H_{mn} = \\langle \\Psi_m | \\hat{H} | \\Psi_n \\rangle,

    and overlap matrix :math:`\\mathbf{S}` with elemets given by

    .. math::
        S_{mn} = \\langle \\Psi_m | \\Psi_n \\rangle,

    both of which can be measured on a quantum device.

    These two matrices then comprise a generalized eigenvalue probelm

    .. math::
        \\mathbf{H}\\mathbf{c} = E \\mathbf{S} \\mathbf{c},

    where E is an approximation to an eigenvalue of the Hamiltonian.

    Attributes
    ----------

    _target_root : int
        Which root of the quantum Krylov subspace should be taken?

    _Ets : float
        The energy for the target state (root).

    _eigenvalues : numpy array
        The quantum subspace eigenvalues.

    _eigenvectors : numpy array
        The quantum subspace eigenvectors.

    _S : numpy array
        The quantum subspace overlap matrix.

    _Hbar : numpy array
        The quantum subspace effective Hamiltonian matrix.

    _Scond : float
        The condition number of the overlap matrix.

    _diagonalize_each_step : bool
        For diagnostic purposes, should the eigenvalue of the target root of the
        quantum Krylov subspace be printed after each new unitary? We recommend
        passing an s so the change in the eigenvalue is small.
    """

    @abstractmethod
    def build_qk_mats(self):
        """Constructs the effective Hamiltonian (:math:`\\mathbf{H}`) and overlap
        (:math:`\\mathbf{S}`) matricies in an efficient maner.

        """
        pass

    def set_circuit_variables(self):
        pass

    def common_run(self):
        # Build S and H matrices
        self._S, self._Hbar = self.build_qk_mats()
        self._Scond = np.linalg.cond(self._S)

        # Get eigenvalues and eigenvectors
        self._eigenvalues, self._eigenvectors = canonical_geig_solve(
            self._S, self._Hbar, print_mats=self._verbose, sort_ret_vals=True
        )

        print(f"\n       ==> {type(self).__name__} eigenvalues <==")
        print("----------------------------------------")
        for i, val in enumerate(self._eigenvalues):
            print("  root  {}  {:.8f}    {:.8f}j".format(i, np.real(val), np.imag(val)))

        # Set ground state energy.
        self._Egs = np.real(self._eigenvalues[0])

        # Set target state energy.
        self._Ets = np.real(self._eigenvalues[self._target_root])

        self.set_circuit_variables()

        # Print summary banner (should done for all algorithms).
        self.print_summary_banner()

        # verify that required attributes were defined
        # (should be called for all algorithms!)
        self.verify_run()

    def get_ts_energy(self):
        """Returns the energy of the target state."""
        return self._Ets

    def get_qk_eigenvalues(self):
        """Returns the vecotor of eigenvalues from the generalized eigenvalue
        problem.
        """
        return self._eigenvalues

    def get_qk_eigenvectors(self):
        """Returns the eigenvectors from the generalized eigenvalue
        problem.
        """
        return self._eigenvectors

    def verify_required_QSD_attributes(self):
        """Verfes that all quantum subspace diagonalizaion attributes are defined
        in concrete class implementations.
        """
        if self._Ets is None:
            raise NotImplementedError(
                "Concrete QK Algorithm class must define self._Ets attribute."
            )

        if self._eigenvalues is None:
            raise NotImplementedError(
                "Concrete QK Algorithm class must define self._eigenvalues attribute."
            )

        if self._S is None:
            raise NotImplementedError(
                "Concrete QK Algorithm class must define self._S attribute."
            )

        if self._Hbar is None:
            raise NotImplementedError(
                "Concrete QK Algorithm class must define self._Hbar attribute."
            )

        if self._Scond is None:
            raise NotImplementedError(
                "Concrete QK Algorithm class must define self._Scond attribute."
            )

        if self._diagonalize_each_step is None:
            raise NotImplementedError(
                "Concrete QK Algorithm class must define self._diagonalize_each_step attribute."
            )
