from abc import abstractmethod
from qforte.abc.algorithm import AnsatzAlgorithm

class PQE(AnsatzAlgorithm):

    @abstractmethod
    def solve(self):
        pass

    def verify_required_PQE_attributes(self):
        if not hasattr(self, '_converged'):
            raise NotImplementedError('Concrete PQE class must define self._converged attribute.')

        if not hasattr(self, '_opt_maxiter'):
            if not hasattr(self, '_diis_maxiter'):
                raise NotImplementedError('Concrete PQE class must define self._diis_maxiter OR self._opt_maxiter attribute.')

        if not hasattr(self, '_opt_thresh'):
            if not hasattr(self, '_res_vec_thresh'):
                raise NotImplementedError('Concrete PQE class must define self._res_vec_thresh OR self._opt_thresh attribute.')
