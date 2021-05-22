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

    def verify_required_UCCPQE_attributes(self):

        if not hasattr(self, '_pool_type'):
            raise NotImplementedError('Concrete UCCVQE class must define self._pool_type attribute.')

        if not hasattr(self, '_pool_obj'):
            raise NotImplementedError('Concrete UCCVQE class must define self._pool_obj attribute.')
