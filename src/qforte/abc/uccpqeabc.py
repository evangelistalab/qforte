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

class UCCPQE(PQE, UCC):
    """The abstract base class inheritied by any algorithm that seeks to find
    eigenstates by minimization of the residual condition

    .. math::
        r_\mu = \langle \Phi_\mu | \hat{U}^\dagger(\mathbf{t}) \hat{H} \hat{U}(\mathbf{t}) | \Phi_0 \\rangle \\rightarrow 0

    using a disentagled UCC type ansatz

    .. math::
        \hat{U}(\mathbf{t}) = \prod_\mu e^{t_\mu (\hat{\\tau}_\mu - \hat{\\tau}_\mu^\dagger)},

    were :math:`\hat{\\tau}_\mu` is a Fermionic excitation operator and
    :math:`t_\mu` is a cluster amplitude.

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

        if not hasattr(self, '_pool_type'):
            raise NotImplementedError('Concrete UCCPQE class must define self._pool_type attribute.')

        if not hasattr(self, '_pool_obj'):
            raise NotImplementedError('Concrete UCCPQE class must define self._pool_obj attribute.')

    #TODO: consider moving functions from uccnpqe or spqe into this class to
    #      to prevent duplication of code

