from abc import abstractmethod
from qforte.abc.algorithm import Algorithm

class QSD(Algorithm):

    @abstractmethod
    def build_qk_mats(self):
        pass

    @abstractmethod
    def build_qk_mats_realistic(self):
        pass

    @abstractmethod
    def build_qk_mats_realistic(self):
        pass

    def get_ts_energy(self):
        return self._Ets

    def get_qk_eigenvalues(self):
        return self._eigenvalues

    def get_qk_eigenvectors(self):
        return self._eigenvectors

    def verify_required_QSD_attributes(self):
        if self._Ets is None:
            raise NotImplementedError('Concrete QK Algorithm class must define self._Ets attribute.')

        if self._eigenvalues is None:
            raise NotImplementedError('Concrete QK Algorithm class must define self._eigenvalues attribute.')

        if self._S is None:
            raise NotImplementedError('Concrete QK Algorithm class must define self._S attribute.')

        if self._Hbar is None:
            raise NotImplementedError('Concrete QK Algorithm class must define self._Hbar attribute.')

        if self._Scond is None:
            raise NotImplementedError('Concrete QK Algorithm class must define self._Scond attribute.')
