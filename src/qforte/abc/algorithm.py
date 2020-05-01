from abc import ABC, abstractmethod
from qforte.utils.state_prep import *

class Algorithm(ABC):
    """
    Attributes
    ----------
    _ref : list
        The set of 1s and 0s indicating the initial quantum state.

    _nqb : int
        The number of qubits the calculation empolys.

    _qb_ham : QuantumOperator
        The operator to be measured (usually the Hamiltonain), mapped to a
        qubit representation.

    _fast : bool
        Whether or not to use a faster version of the algorithm that bypasses
        measurment (unphysical for quantum computer).

    _trotter_order : int
        The Trotter order to use for exponentiated operators.
        (exact in the infinte limit).

    _trotter_number : int
        The Trotter number (or the number of trotter steps)
        to use for exponentiated operators.
        (exact in the infinte limit).

    _Egs : float
        The final ground state energy value.

    _Umaxdepth : QuantumCircuit
        The deepest circuit used during any part of the algorithm.

    _n_ham_measurements : int
        The total number of times the energy was evaluated via
        measurement of the Hamiltoanin



    Methods
    -------
    build_Uprep()
        Returns a QuantumCircuit object corresponding to the state preparation
        circuit reference state (usually a small product of X gates).


    """

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

#         if self._n_ham_measurements is None:
#             raise NotImplementedError('Concrete Algorithm class must define self._n_ham_measurements attribute.')

#         if self._n_ham_measurements is None:
#             raise NotImplementedError('Concrete Algorithm class must define self._n_ham_measurements attribute.')
