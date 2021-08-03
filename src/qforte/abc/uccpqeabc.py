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
from qforte.utils.op_pools import *

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

    _diis_maxiter : int
        The maximum number of DIIS iteratios to perform.

    _res_vec_thresh : float
        The numberical threshold for the norm of the residual vector, used to
        determine when PQE has converged.

    _res_vec_evals : int
        The total number of times the entire residual was evaluated.

    _res_m_evals : int
        The total number of times an individal residual element was evaluated.

    _orb_e : list floats
        The Hartree-Fock orbital energies

    """

    def verify_required_UCCPQE_attributes(self):

        if not hasattr(self, '_pool_type'):
            raise NotImplementedError('Concrete UCCVQE class must define self._pool_type attribute.')

        if not hasattr(self, '_pool_obj'):
            raise NotImplementedError('Concrete UCCVQE class must define self._pool_obj attribute.')

    #TODO: consider moving functions from uccnpqe or spqe into this class to
    #      to prevent duplication of code

    def get_res_over_mpdenom(self, residuals):
        """This function returns a vector given by the residuals dividied by the
        respective Moller Plesset denominators.

        Parameters
        ----------
        residuals : list of floats
            The list of (real) floating point numbers which represent the
            residuals.
        """

        resids_over_denoms = []

        # each operator needs a score, so loop over toperators
        for mu, m in enumerate(self._tops):
            sq_op = self._pool[m][1]

            temp_idx = sq_op.terms()[0][2][-1]
            if temp_idx < int(sum(self._ref)/2): # if temp_idx is an occupid idx
                sq_creators = sq_op.terms()[0][1]
                sq_annihilators = sq_op.terms()[0][2]
            else:
                sq_creators = sq_op.terms()[0][2]
                sq_annihilators = sq_op.terms()[0][1]

            destroyed = False
            denom = 0.0

            denom = sum(self._orb_e[x] for x in sq_annihilators) - sum(self._orb_e[x] for x in sq_creators)

            res_mu = copy.deepcopy(residuals[mu])
            res_mu /= denom # divide by energy denominator

            resids_over_denoms.append(res_mu)

        return resids_over_denoms

