from abc import abstractmethod
from qforte.abc.vqeabc import VQE
from qforte.utils.op_pools import *

class UCCVQE(VQE):

    # TODO (cleanup): rename functin 'build_pool'
    def fill_pool(self):
        if (self._pool_type=='SD'):
            self._pool_obj = SDOpPool(self._ref)
        else:
            raise ValueError('Invalid operator pool type specified.')

        self._pool_obj.fill_pool()
        # TODO (opt): rename _pool to '_pool_lst' and conver to numpy array.
        self._pool = self._pool_obj.get_pool_lst()

    def verify_required_UCCVQE_attributes(self):
        if self._use_analytic_grad is None:
            raise NotImplementedError('Concrete UCCVQE class must define self._use_analytic_grad attribute.')

        if self._pool_type is None:
            raise NotImplementedError('Concrete UCCVQE class must define self._pool_type attribute.')

        if self._pool_obj is None:
            raise NotImplementedError('Concrete UCCVQE class must define self._pool_obj attribute.')
