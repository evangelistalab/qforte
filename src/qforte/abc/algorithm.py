from abc import ABC, abstractmethod
from qforte.utils.state_prep import *

class Algorithm(ABC):

    def __init__(self,
                 system,
                 reference,
                 trial_state_type='reference',
                 trotter_order=1,
                 trotter_number=1,
                 fast=True,
                 verbose=False):

        self._sys = system
        self._ref = reference
        self._nqb = len(reference)
        self._trial_state_type = trial_state_type
        self._Uprep = build_Uprep(reference, trial_state_type)
        # TODO (Nick): change Molecule.get_hamiltonian() to Molecule.get_qb_hamiltonain()
        self._qb_ham = system.get_hamiltonian()
        self._trotter_order = trotter_order
        self._trotter_number = trotter_number
        self._fast = fast
        self._verbose = verbose

        # Required attributes, to be defined in concrete class.
        self._Egs = None
        self._Umaxdepth = None
        self._tot_Nmeasurements = None
        self._tot_Npreps = None

    @abstractmethod
    def print_options_banner(self):
        pass

    @abstractmethod
    def print_summary_banner(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def run_realistic(self):
        pass

    @abstractmethod
    def verify_run(self):
        pass

    def get_gs_energy(self):
        return self._Egs

    def get_Umaxdepth(self):
        pass

    def get_tot_measurements(self):
        pass

    def get_tot_state_preparations(self):
        pass

    def verify_required_attributes(self):
        if self._Egs is None:
            raise NotImplementedError('Concrete Algorithm class must define self._Egs attribute.')

#         if self._Umaxdepth is None:
#             raise NotImplementedError('Concrete Algorithm class must define self._Umaxdepth attribute.')

#         if self._tot_Nmeasurements is None:
#             raise NotImplementedError('Concrete Algorithm class must define self._tot_Nmeasurements attribute.')

#         if self._tot_Npreps is None:
#             raise NotImplementedError('Concrete Algorithm class must define self._tot_Npreps attribute.')
        
