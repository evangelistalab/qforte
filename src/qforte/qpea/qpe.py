import qforte
from qforte.abc.algorithm import Algorithm
from qforte.abc.mixin import Trotterizable
from qforte.utils.transforms import (
    circuit_to_organizer,
    organizer_to_circuit,
    join_organizers,
    get_jw_organizer,
)

from qforte.utils.state_prep import *
from qforte.utils.trotterization import trotterize, trotterize_w_cRz


import numpy as np
from scipy import stats


class QPE(Trotterizable, Algorithm):
    def run(
        self, guess_energy: float, t=1.0, nruns=20, success_prob=0.5, num_precise_bits=4
    ):
        """
        guess_energy : A guess for the eigenvalue of the eigenspace with which |0>^(n)
            has greatest overlap. You should be confident the ground state is within
        t : A scaling parameter that controls the precision of the computation. You should be
            confident that the eigenvalue of interest is within +/- 2pi/t of the guess energy.
            Larger t's lead to fewer resources for the same amount of precision, but require
            more confidence in the guess energy.
        """

        # float: evolution times
        self._t = t
        # int: number of times to sample the eigenvalue distribution
        self._nruns = nruns
        self._success_prob = success_prob
        self._num_precise_bits = num_precise_bits
        self._Uqpe = qforte.Circuit()
        # int: The number of qubits needed to represent the state.
        self._n_state_qubits = self._nqb
        eps = 1 - success_prob
        # int: The number of ancilla qubits used to hold eigenvalue information
        self._n_ancilla = num_precise_bits + int(np.log2(2 + (1.0 / eps)))
        # int: The total number of qubits needed in the circuit
        self._n_tot_qubits = self._n_state_qubits + self._n_ancilla
        self._abegin = self._n_state_qubits
        self._aend = self._n_tot_qubits - 1

        self._n_classical_params = 0
        self._n_pauli_trm_measures = nruns

        self._guess_energy = guess_energy
        self._guess_periods = round(self._t * guess_energy / (-2 * np.pi))

        # Print options banner (should done for all algorithms).
        self.print_options_banner()

        ######### QPE ########

        # Apply Hadamard gates to all ancilla qubits
        self._Uqpe.add(self.get_Uhad())

        # Prepare the trial state on the non-ancilla qubits
        self._Uqpe.add(self._Uprep)

        # add controll e^-iHt circuit
        self._Uqpe.add(self.get_dynamics_circ())

        # add reverse QFT
        self._Uqpe.add(self.get_qft_circuit("reverse"))

        computer = qforte.Computer(self._n_tot_qubits)
        computer.apply_circuit(self._Uqpe)

        self._n_cnot = self._Uqpe.get_num_cnots()

        if self._fast:
            z_readouts = computer.measure_z_readouts_fast(
                self._abegin, self._aend, self._nruns
            )
        else:
            Zcirc = self.get_z_circuit()
            z_readouts = computer.measure_readouts(Zcirc, self._nruns)

        self._phases = []
        for readout in z_readouts:
            val = sum(z / (2**i) for i, z in enumerate(readout, start=1))
            self._phases.append(val)

        # find final binary string of phase readouts:
        final_readout = []
        final_readout_aves = []
        for i in range(self._n_ancilla):
            iave = sum(readout[i] for readout in z_readouts) / nruns
            final_readout_aves.append(iave)
            final_readout.append(1 if iave > 0.5 else 0)

        self._final_phase = sum(
            z / (2**i) for i, z in enumerate(final_readout, start=1)
        )

        E_u = -2 * np.pi * (self._final_phase + self._guess_periods - 1) / t
        E_l = -2 * np.pi * (self._final_phase + self._guess_periods - 0) / t
        E_qpe = E_l if abs(E_l - guess_energy) < abs(E_u - guess_energy) else E_u

        res = stats.mode(np.asarray(self._phases))
        self._mode_phase = res.mode
        E_u = -2 * np.pi * (self._mode_phase + self._guess_periods - 1) / t
        E_l = -2 * np.pi * (self._mode_phase + self._guess_periods - 0) / t
        self._mode_energy = (
            E_l if abs(E_l - guess_energy) < abs(E_u - guess_energy) else E_u
        )

        print("\n           ==> QPE readout averages <==")
        print("------------------------------------------------")
        for i, ave in enumerate(final_readout_aves):
            print("  bit ", i, ": ", ave)
        print("\n  Final bit readout: ", final_readout)

        ######### QPE ########

        # set Egs
        self._Egs = E_qpe

        # set Umaxdepth
        self._Umaxdepth = self._Uqpe

        # Print summary banner (should done for all algorithms).
        self.print_summary_banner()

        # verify that required attributes were defined
        # (should done for all algorithms).
        self.verify_run()

    def run_realistic(self):
        raise NotImplementedError(
            "run_realistic() for QPE can be done by initializing QPE with fast==False and calling run()."
        )

    def verify_run(self):
        self.verify_required_attributes()

    def print_options_banner(self):
        print("\n-----------------------------------------------------")
        print("       Quantum Phase Estimation Algorithm   ")
        print("-----------------------------------------------------")

        print("\n\n                 ==> QPE options <==")
        print("-----------------------------------------------------------")
        # General algorithm options.
        self.print_generic_options()

        # Specific QPE options.
        print("Target success probability:              ", self._success_prob)
        print("Number of precise bits for phase:        ", self._num_precise_bits)
        print("Number of time steps:                    ", self._n_ancilla)
        print("Evolution time (t):                      ", self._t)
        print("Number of QPE algorithm executions:      ", self._nruns)
        print("\n")

    def print_summary_banner(self):
        print("\n\n                        ==> QPE summary <==")
        print("---------------------------------------------------------------")
        print("Final QPE Energy:                        ", np.round(self._Egs, 10))
        print(
            "Mode QPE Energy:                         ", np.round(self._mode_energy, 10)
        )
        print(
            "Final QPE phase:                          ",
            np.round(self._final_phase, 10),
        )
        print(
            "Mode QPE phase:                           ", np.round(self._mode_phase, 10)
        )
        print("Number of classical parameters used:      ", self._n_classical_params)
        print("Number of CNOT gates in deepest circuit:  ", self._n_cnot)
        print("Number of Pauli term measurements:        ", self._n_pauli_trm_measures)

    ### QPE specific methods

    def get_Uhad(self):
        """Generates a circuit which to puts all of the ancilla regester in
        superpostion.

            Arguments
            ---------

            self._abegin : int
                The index of the begin qubit.

            self._aend : int
                The index of the end qubit.

            Returns
            -------

            qft_circ : Circuit
                A circuit of consecutive Hadamard gates.
        """
        Uhad = qforte.Circuit()
        for j in range(self._abegin, self._aend + 1):
            Uhad.add(qforte.gate("H", j))

        return Uhad

    def get_dynamics_circ(self):
        """Generates controlled unitaries. Ancilla qubit n controls a Trotter
        approximation to exp(-iHt*2^n).

        Returns
        -------
        U : Circuit
            A circuit approximating controlled application of e^-iHt.
        """
        U = qforte.Circuit()
        ancilla_idx = self._abegin

        temp_op = qforte.QubitOperator()
        scalar_terms = []
        for scalar, operator in self._qb_ham.terms():
            phase = -1.0j * scalar * self._t
            if operator.size() == 0:
                scalar_terms.append(scalar * self._t)
            else:
                # Strangely, the code seems to work even if this line is outside the else clause.
                # TODO: Figure out how.
                temp_op.add(phase, operator)

        for n in range(self._n_ancilla):
            tn = 2**n
            expn_op, _ = trotterize_w_cRz(
                temp_op, ancilla_idx, trotter_number=self._trotter_number
            )

            # Rotation for the scalar Hamiltonian term
            U.add(
                qforte.gate(
                    "R",
                    ancilla_idx,
                    ancilla_idx,
                    -1.0 * np.sum(scalar_terms) * float(tn),
                )
            )

            for i in range(tn):
                U.add_circuit(expn_op)

            ancilla_idx += 1

        return U

    def get_qft_circuit(self, direct):
        """Generates a circuit for Quantum Fourier Transformation with no swapping
        of bit positions.

            Arguments
            ---------

            self._abegin : int
                The index of the begin qubit.

            self._aend : int
                The index of the end qubit.

            direct : string
                The direction of the Fourier Transform can be 'forward' or 'reverse.'

            Returns
            -------

            qft_circ : Circuit
                A circuit representing the Quantum Fourier Transform.
        """

        qft_circ = qforte.Circuit()
        lens = self._aend - self._abegin + 1
        for j in range(lens):
            qft_circ.add(qforte.gate("H", j + self._abegin))
            for k in range(2, lens + 1 - j):
                phase = 2.0 * np.pi / (2**k)
                qft_circ.add(
                    qforte.gate("cR", j + self._abegin, j + k - 1 + self._abegin, phase)
                )

        if direct == "forward":
            return qft_circ
        elif direct == "reverse":
            return qft_circ.adjoint()
        else:
            raise ValueError('QFT directions can only be "forward" or "reverse"')

    def get_z_circuit(self):
        """Generates a circuit of Z gates for each quibit in the ancilla register.

        Arguments
        ---------

        self._abegin : int
            The index of the begin qubit.

        self._aend : int
            The index of the end qubit.

        Returns
        -------

        z_circ : Circuit
            A circuit representing the the Z gates to be measured.
        """

        Z_circ = qforte.Circuit()
        for j in range(self._abegin, self._aend + 1):
            Z_circ.add(qforte.gate("Z", j))

        return Z_circ
