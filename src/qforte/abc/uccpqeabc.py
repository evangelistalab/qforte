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

    def fill_pool(self):

        self._pool_obj = qf.SQOpPool()
        self._pool_obj.set_orb_spaces(self._ref)

        if self._pool_type in {'sa_SD', 'GSD', 'SD', 'SDT', 'SDTQ', 'SDTQP', 'SDTQPH'}:
            self._pool_obj.fill_pool(self._pool_type)
        else:
            raise ValueError('Invalid operator pool type specified.')

        self._pool = self._pool_obj.terms()

        self._grad_vec_evals = 0
        self._grad_m_evals = 0
        self._k_counter = 0
        self._grad_m_evals = 0
        self._prev_energy = self._hf_energy
        self._curr_energy = 0.0
        self._curr_grad_norm = 0.0

        self._Nm = [len(operator.jw_transform().terms()) for _, operator in self._pool_obj.terms()]

    def verify_required_UCCPQE_attributes(self):

        if not hasattr(self, '_pool_type'):
            raise NotImplementedError('Concrete UCCVQE class must define self._pool_type attribute.')

        if not hasattr(self, '_pool_obj'):
            raise NotImplementedError('Concrete UCCVQE class must define self._pool_obj attribute.')
