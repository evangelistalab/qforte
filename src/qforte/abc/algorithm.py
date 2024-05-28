"""
Algorithm and AnsatzAlgorithm base classes
==========================================
The abstract base classes inherited by all algorithm subclasses.
"""

from abc import ABC, abstractmethod
import qforte as qf
from qforte.utils.state_prep import *
from qforte.abc.mixin import Trotterizable


class Algorithm(ABC):
    """A class that characterizes the most basic functionality for all
    other algorithms.

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
        measurment (unphysical for quantum computer). Most algorithms only
        have a fast implentation.

    _Egs : float
        The final ground state energy value.

    _Umaxdepth : QuantumCircuit
        The deepest circuit used during any part of the algorithm.

    _n_classical_params : int
        The number of classical parameters used by the algorithm.

    _n_cnot : int
        The number of controlled-not (CNOT) opperations used in the (deepest)
        quantum circuit (_Umaxdepth).

    _n_pauli_trm_measures : int
        The number of pauli terms (Hermitian products of Pauli X, Y, and/or Z gates)
        mesaured over the entire algorithm.

    _res_vec_evals : int
        The total number of times the entire residual was evaluated.

    _res_m_evals : int
        The total number of times an individual residual element was evaluated.
    """

    def __init__(
        self,
        system,
        reference=None,
        state_prep_type="occupation_list",
        fast=True,
        verbose=False,
        print_summary_file=False,
        **kwargs,
    ):
        if isinstance(self, qf.QPE) and hasattr(system, "frozen_core"):
            if system.frozen_core + system.frozen_virtual > 0:
                raise ValueError("QPE with frozen orbitals is not currently supported.")

        self._sys = system
        self._state_prep_type = state_prep_type

        if self._state_prep_type == "occupation_list":
            if reference is None:
                self._ref = system.hf_reference
            else:
                if not (isinstance(reference, list)):
                    raise ValueError(
                        "occupation_list reference must be list of 1s and 0s."
                    )
                self._ref = reference

            self._refprep = build_refprep(self._ref)
            self._Uprep = qf.Circuit(self._refprep)

        elif self._state_prep_type == "unitary_circ":
            if not isinstance(reference, qf.Circuit):
                raise ValueError("unitary_circ reference must be a Circuit.")

            self._ref = system.hf_reference
            self._refprep = build_refprep(self._ref)
            self._Uprep = reference

        elif self._state_prep_type == "computer":
            if not isinstance(reference, qf.Computer):
                raise ValueError("computer reference must be a Computer.")
            if not fast:
                raise ValueError(
                    "`self._fast = False` specifies not to skip steps, but `self._state_prep_type = computer` specifies to skip state initialization. That's inconsistent."
                )
            if reference.get_nqubit() != len(system.hf_reference):
                raise ValueError(
                    f"Computer needs {len(system.hf_reference)} qubits, found {reference.get_nqubit()}."
                )
            if (
                not hasattr(self, "computer_initializable")
                or not self.computer_initializable
            ):
                raise ValueError("Class cannot be initialized with a computer.")

            self._ref = system.hf_reference
            self._refprep = build_refprep(self._ref)
            self._Uprep = qf.Circuit()
            self.computer = reference

        else:
            raise ValueError(
                "QForte only supports references as occupation lists, Circuits, or Computers."
            )

        self._nqb = len(self._ref)
        self._qb_ham = system.hamiltonian
        if self._qb_ham.num_qubits() != self._nqb:
            raise ValueError(
                f"The reference has {self._nqb} qubits, but the Hamiltonian has {self._qb_ham.num_qubits()}. This is inconsistent."
            )
        try:
            self._hf_energy = system.hf_energy
        except AttributeError:
            self._hf_energy = 0.0

        self._Nl = len(self._qb_ham.terms())

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
        """Prints the run options used for algorithm."""
        pass

    @abstractmethod
    def print_summary_banner(self):
        """Prints a summary of the post-run information."""
        pass

    @abstractmethod
    def run(self):
        """Executes the algorithm."""
        pass

    @abstractmethod
    def run_realistic(self):
        """Executes the algorithm using only operations physically possible for
        quantum hardware. Not implemented for most algorithms.
        """
        pass

    @abstractmethod
    def verify_run(self):
        """Verifies that the abstract sub-class(es) define the required attributes."""
        pass

    def get_gs_energy(self):
        """Returns the final ground state energy."""
        return self._Egs

    def get_Umaxdepth(self):
        """Returns the deepest circuit used during any part of the
        algorithm (_Umaxdepth).
        """
        pass

    def get_tot_measurements(self):
        pass

    def get_tot_state_preparations(self):
        pass

    def verify_required_attributes(self):
        """Verifies that the concrete sub-class(es) define the required attributes."""
        if self._Egs is None:
            raise NotImplementedError(
                "Concrete Algorithm class must define self._Egs attribute."
            )

        #         if self._Umaxdepth is None:
        #             raise NotImplementedError('Concrete Algorithm class must define self._Umaxdepth attribute.')

        if self._n_classical_params is None:
            raise NotImplementedError(
                "Concrete Algorithm class must define self._n_classical_params attribute."
            )

        if self._n_cnot is None:
            raise NotImplementedError(
                "Concrete Algorithm class must define self._n_cnot attribute."
            )

        if self._n_pauli_trm_measures is None:
            raise NotImplementedError(
                "Concrete Algorithm class must define self._n_pauli_trm_measures attribute."
            )

    def print_generic_options(self):
        """Print options applicable to any algorithm."""
        print(
            "Trial reference state:                   ",
            ref_string(self._ref, self._nqb),
        )
        print("Number of Hamiltonian Pauli terms:       ", self._Nl)
        print("Trial state preparation method:          ", self._state_prep_type)
        if isinstance(self, Trotterizable):
            self.print_trotter_options()
        print("Use fast version of algorithm:           ", str(self._fast))
        if not self._fast:
            print("Measurement variance thresh:             ", 0.01)


class AnsatzAlgorithm(Algorithm):
    """A class that characterizes the most basic functionality for all
    other algorithms which utilize an operator ansatz such as VQE.

    Attributes
    ----------
    _curr_energy: float
        The energy at the current iteration step.

    _Nm: list of int
        Number of circuits for each operator in the pool.

    _opt_maxiter : int
        The maximum number of iterations for the classical optimizer.

    _opt_thresh : float
        The numerical convergence threshold for the specified classical
        optimization algorithm. Is usually the norm of the gradient, but
        is algorithm dependant, see scipy.minimize.optimize for details.

    _pool_obj : SQOpPool
        A pool of second quantized operators we use in the ansatz.

    _tops : list
        A list of indices representing selected operators in the pool.

    _tamps : list
        A list of amplitudes (to be optimized) representing selected
        operators in the pool.

    _qubit_excitations: bool
        Controls the use of qubit/fermionic excitations.

    _compact_excitations: bool
        Controls the use of compact quantum circuits for fermion/qubit
        excitations.
    """

    # TODO (opt major): write a C function that prepares this super efficiently
    def build_Uvqc(self, amplitudes=None):
        """This function returns the Circuit object built
        from the appropriate amplitudes (tops)

        Parameters
        ----------
        amplitudes : list
            A list of parameters that define the variational degrees of freedom in
            the state preparation circuit Uvqc. This is needed for the scipy minimizer.
        """

        U = self.ansatz_circuit(amplitudes)

        Uvqc = qforte.Circuit()
        Uvqc.add(self._Uprep)
        Uvqc.add(U)

        return Uvqc

    def fill_pool(self):
        """This function populates an operator pool with SQOperator objects."""

        if self._pool_type in {"sa_SD", "GSD", "SD", "SDT", "SDTQ", "SDTQP", "SDTQPH"}:
            self._pool_obj = qf.SQOpPool()
            if hasattr(self._sys, "orb_irreps_to_int"):
                self._pool_obj.set_orb_spaces(self._ref, self._sys.orb_irreps_to_int)
            else:
                self._pool_obj.set_orb_spaces(self._ref)
            self._pool_obj.fill_pool(self._pool_type)
        elif isinstance(self._pool_type, qf.SQOpPool):
            self._pool_obj = self._pool_type
        else:
            raise ValueError("Invalid operator pool type specified.")

        self._Nm = [
            len(operator.jw_transform().terms()) for _, operator in self._pool_obj
        ]

    def measure_energy(self, Ucirc, computer=None):
        """
        This function returns the energy expectation value of the state
        Ucirc|Ψ>.

        Parameters
        ----------
        Ucirc : Circuit
            The state preparation circuit.
        """
        if self._fast:
            if computer is None:
                computer = qf.Computer(self._nqb)
            computer.apply_circuit(Ucirc)
            val = np.real(computer.direct_op_exp_val(self._qb_ham))
        else:
            if compute is not None:
                raise TypeError(
                    "measure_energy in slow mode does not support custom Computer."
                )
            Exp = qforte.Experiment(self._nqb, Ucirc, self._qb_ham, 2000)
            val = Exp.perfect_experimental_avg()

        assert np.isclose(np.imag(val), 0.0)

        return val

    def __init__(
        self,
        *args,
        qubit_excitations=False,
        compact_excitations=False,
        diis_max_dim=8,
        max_moment_rank=0,
        moment_dt=None,
        penalty=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._curr_energy = 0
        self._Nm = []
        self._tamps = []
        self._tops = []
        self._pool_obj = qf.SQOpPool()
        self._qubit_excitations = qubit_excitations
        self._compact_excitations = compact_excitations
        self._diis_max_dim = diis_max_dim
        # The max_moment_rank controls the calculation of non-iterative energy corrections
        # based on the method of moments of coupled-cluster theory.
        # max_moment_rank = 0: non-iterative correction skipped
        # max_moment_rank = n: projections up to n-tuply excited Slater determinants are considered
        self._max_moment_rank = max_moment_rank
        # The moment_dt variable defines the 'residual' state used to measure the residuals for the moment corrections
        self._moment_dt = moment_dt
        self._penalty = penalty

        if self._penalty is not None:
            if isinstance(self, qf.UCCNPQE) or isinstance(self, qf.SPQE):
                raise ValueError(
                    "PQE with Hamiltonian penalty terms not yet supported."
                )
            expected_keys = {"operators", "eigenvalues", "scaling_factors"}
            if not isinstance(self._penalty, dict):
                raise ValueError(
                    f"The 'penalty' option must be a dictionary with keys: {expected_keys}"
                )
            if not set(self._penalty.keys()) == expected_keys:
                raise ValueError(
                    f"Incorrect keys in 'penalty' dictionary. Expected keys: {expected_keys}"
                )
            if not all(isinstance(value, list) for value in self._penalty.values()):
                raise ValueError(
                    "All values in the 'penalty' dictionary must be lists."
                )
            if (
                not len(self._penalty["operators"])
                == len(self._penalty["eigenvalues"])
                == len(self._penalty["scaling_factors"])
            ):
                raise ValueError(
                    "Operators, eigenvalues, and scaling factors lists must be of the same length."
                )
            if not all(
                isinstance(op, qf.QubitOperator)
                for op in self._penalty.get("operators", [])
            ):
                raise ValueError(
                    "All elements in 'operators' must be instances of the QubitOperator class."
                )
            if not all(
                isinstance(eigval, (int, float))
                for eigval in self._penalty.get("eigenvalues", [])
            ):
                raise ValueError("All elements in 'eigenvalues' must be real numbers.")
            if not all(
                isinstance(scaling, (int, float))
                for scaling in self._penalty.get("scaling_factors", [])
            ):
                raise ValueError(
                    "All elements in 'scaling_factors' must be real numbers."
                )
            penalties_qop = qf.QubitOperator()
            for i in range(len(self._penalty["eigenvalues"])):
                eig = qf.Circuit()
                temp_qop = qf.QubitOperator()
                penalty_qop = qf.QubitOperator()
                temp_qop.add(self._penalty["operators"][i])
                temp_qop.add(-self._penalty["eigenvalues"][i], eig)
                penalty_qop.add(temp_qop)
                penalty_qop.operator_product(temp_qop, True, True)
                penalty_qop.mult_coeffs(self._penalty["scaling_factors"][i])
                penalties_qop.add(penalty_qop)
                penalties_qop.simplify(True)
            self._qb_ham = qf.QubitOperator()
            self._qb_ham.add(self._sys.hamiltonian)
            self._qb_ham.add(penalties_qop)
            self._qb_ham.simplify(True)
            self._Nl = len(self._qb_ham.terms())

        kwargs.setdefault("irrep", None)
        if hasattr(self._sys, "point_group"):
            irreps = list(range(len(self._sys.point_group[1])))
            if kwargs["irrep"] is None:
                print(
                    "\nWARNING: The {0} point group was detected, but no irreducible representation was specified.\n"
                    "         Proceeding with totally symmetric.\n".format(
                        self._sys.point_group[0].capitalize()
                    )
                )
                self._irrep = 0
            elif kwargs["irrep"] in irreps:
                self._irrep = kwargs["irrep"]
            else:
                raise ValueError(
                    "{0} is not an irreducible representation of {1}.\n"
                    "               Choose one of {2} corresponding to the\n"
                    "               {3} irreducible representations of {1}".format(
                        kwargs["irrep"],
                        self._sys.point_group[0].capitalize(),
                        irreps,
                        self._sys.point_group[1],
                    )
                )
        elif kwargs["irrep"] is not None:
            print(
                "\nWARNING: Point group information not found.\n"
                '         Ignoring "irrep" and proceeding without symmetry.\n'
            )

    def energy_feval(self, params):
        """
        This function returns the energy expectation value of the state
        Uprep(params)|Ψ>, where params are parameters that can be optimized
        for some purpouse such as energy minimization.

        Parameters
        ----------
        params : list of floats
            The dist of (real) floating point number which will characterize
            the state preparation circuit.
        """
        Ucirc = self.build_Uvqc(amplitudes=params)
        Energy = self.measure_energy(Ucirc, self.get_initial_computer())

        self._curr_energy = Energy
        return Energy

    def get_initial_computer(self) -> qf.Computer:
        if hasattr(self, "computer"):
            return qf.Computer(self.computer)
        else:
            return qf.Computer(self._nqb)
