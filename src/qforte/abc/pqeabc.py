from abc import abstractmethod
from qforte.abc.algorithm import Algorithm

class PQE(Algorithm):

    @abstractmethod
    def build_Uvqc(self):
        pass

    @abstractmethod
    def measure_energy(self):
        pass

    @abstractmethod
    def energy_feval(self):
        pass

    @abstractmethod
    def solve(self):
        pass

    def verify_required_PQE_attributes(self):
        # if not hasattr(self, '_optimizer'):
        #     raise NotImplementedError('Concrete PQE class must define self._optimizer attribute.')

        if not hasattr(self, '_converged'):
            raise NotImplementedError('Concrete PQE class must define self._converged attribute.')

        # if not hasattr(self, '_final_result'):
        #     raise NotImplementedError('Concrete PQE class must define self._final_result attribute.')

        # if self._opt_maxiter is None:
        #     if self._diis_maxiter is None:
        #         raise NotImplementedError('Concrete PQE class must define self._diis_maxiter OR self._opt_maxiter attribute.')

        if not hasattr(self, '_opt_maxiter'):
            if not hasattr(self, '_diis_maxiter'):
                raise NotImplementedError('Concrete PQE class must define self._diis_maxiter OR self._opt_maxiter attribute.')

        if not hasattr(self, '_opt_thresh'):
            if not hasattr(self, '_res_vec_thresh'):
                raise NotImplementedError('Concrete PQE class must define self._res_vec_thresh OR self._opt_thresh attribute.')
