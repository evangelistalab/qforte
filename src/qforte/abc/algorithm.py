from abc import ABC, abstractmethod
import qforte as qf
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
        The operator to be measured (usually the Hamiltonian), mapped to a
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
        measurement of the Hamiltonian



    Methods
    -------
    build_Uprep()
        Returns a QuantumCircuit object corresponding to the state preparation
        circuit reference state (usually a small product of X gates).


    """

    def __init__(self,
                 system,
                 reference=None,
                 trial_state_type='occupation_list',
                 trotter_order=1,
                 trotter_number=1,
                 fast=True,
                 verbose=False,
                 print_summary_file=False):

        self._sys = system
        self._trial_state_type = trial_state_type

        if self._trial_state_type == 'occupation_list':
            if(reference==None):
                self._ref = system.get_hf_reference()
            else:
                if not (isinstance(reference, list)):
                    raise ValueError("occupation_list reference must be list of 1s and 0s.")
                self._ref = reference

            self._Uprep = build_Uprep(self._ref, trial_state_type)

        elif self._trial_state_type == 'unitary_circ':
            if(reference==None):
                if not (isinstance(reference, qf.QuantumCircuit)):
                    raise ValueError("unitary_circ reference must be a QuantumCircuit.")

            else:
                self._ref = system.get_hf_reference()
                self._Uprep = reference

        else:
            raise ValueError("QForte only suppors references as occupation lists and QuantumCircuits.")


        self._nqb = len(self._ref)

        # TODO (Nick): change Molecule.get_hamiltonian() to Molecule.get_qb_hamiltonian()
        self._qb_ham = system.get_hamiltonian()
        if self._qb_ham.num_qubits() != self._nqb:
            raise ValueError(f"The reference has {self._nqb} qubits, but the Hamiltonian has {self._qb_ham.num_qubits()}. This is inconsistent.")
        if hasattr(system, '_hf_energy'):
            self._hf_energy = system.get_hf_energy()
        else:
            self._hf_energy = 0.0

        self._Nl = len(self._qb_ham.terms())
        self._trotter_order = trotter_order
        self._trotter_number = trotter_number
        self._fast = fast
        self._verbose = verbose
        self._print_summary_file = print_summary_file

        self._noise_factor = 0.0

        # Required attributes, to be defined in concrete class.
        self._Egs = None
        self._Umaxdepth = None
        self._n_classical_params = None
        self._n_cnot = None
        self._n_pauli_trm_measures = None


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

        if self._n_classical_params is None:
            raise NotImplementedError('Concrete Algorithm class must define self._n_classical_params attribute.')

        if self._n_cnot is None:
            raise NotImplementedError('Concrete Algorithm class must define self._n_cnot attribute.')

        if self._n_pauli_trm_measures is None:
            raise NotImplementedError('Concrete Algorithm class must define self._n_pauli_trm_measures attribute.')

class AnsatzAlgorithm(Algorithm):
    """
    Attributes
    ----------
    _curr_energy: float
        The energy at the current iteration step.

    _Nm: list of int
        Number of circuits for each operator in the pool.

    _pool : list of tuple(complex, SqOperator)
        The linear combination of (optionally symmetrized) single and double
        excitation operators to consider. This is represented as a list.
        Each entry is a pair of a complex coefficient and an SqOperator object.

    _pool_obj : SqOpPool
        A pool of second quantized operators we use in the ansatz.

    _tops : list
        A list of indices representing selected operators in the pool.

    _tamps : list
        A list of amplitudes (to be optimized) representing selected
        operators in the pool.
    """

    @abstractmethod
    def ansatz_circuit(self):
        pass

    # TODO (opt major): write a C function that prepares this super efficiently
    def build_Uvqc(self, amplitudes=None):
        """ This function returns the QuantumCircuit object built
        from the appropriate amplitudes (tops)

        Parameters
        ----------
        amplitudes : list
            A list of parameters that define the variational degrees of freedom in
            the state preparation circuit Uvqc. This is needed for the scipy minimizer.
        """

        U = self.ansatz_circuit(amplitudes)

        Uvqc = qforte.QuantumCircuit()
        Uvqc.add_circuit(self._Uprep)
        Uvqc.add_circuit(U)

        return Uvqc

    def fill_pool(self):

        self._pool_obj = qf.SQOpPool()
        self._pool_obj.set_orb_spaces(self._ref)

        if self._pool_type in {'sa_SD', 'GSD', 'SD', 'SDT', 'SDTQ', 'SDTQP', 'SDTQPH'}:
            self._pool_obj.fill_pool(self._pool_type)
        else:
            raise ValueError('Invalid operator pool type specified.')

        self._pool = self._pool_obj.terms()

        self._Nm = [len(operator.jw_transform().terms()) for _, operator in self._pool]

    def measure_energy(self, Ucirc):
        """
        Parameters
        ----------
        Ucirc : QuantumCircuit
            The state preparation circuit.
        """
        if self._fast:
            myQC = qforte.QuantumComputer(self._nqb)
            myQC.apply_circuit(Ucirc)
            val = np.real(myQC.direct_op_exp_val(self._qb_ham))
        else:
            Exp = qforte.Experiment(self._nqb, Ucirc, self._qb_ham, 2000)
            val = Exp.perfect_experimental_avg([])

        assert np.isclose(np.imag(val), 0.0)

        return val

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._curr_energy = 0
        self._Nm = []
        self._tamps = []
        self._tops = []
        self._pool = []
        self._pool_obj = qf.SQOpPool()

    def energy_feval(self, params):
        Ucirc = self.build_Uvqc(amplitudes=params)
        Energy = self.measure_energy(Ucirc)

        self._curr_energy = Energy
        return Energy
