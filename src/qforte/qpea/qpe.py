import qforte
from qforte.abc.algorithm import Algorithm
from qforte.utils.transforms import (circuit_to_organizer,
                                    organizer_to_circuit,
                                    join_organizers,
                                    get_jw_organizer)

from qforte.utils.state_prep import *
from qforte.utils.trotterization import (trotterize,
                                         trotterize_w_cRz)


import numpy as np
from scipy import stats

class QPE(Algorithm):
    def run(self,
            t = 1.0,
            nruns = 20,
            success_prob = 0.5,
            num_precise_bits = 4,
            return_phases=False):

        self._t = t
        self._nruns = nruns
        self._success_prob = success_prob
        self._num_precise_bits = num_precise_bits
        self._return_phases = return_phases
        self._Uqpe = qforte.QuantumCircuit()
        self._n_state_qubits = self._nqb
        eps = 1 - success_prob
        self._n_ancilla = num_precise_bits + int(np.log2(2 + (1.0/eps)))
        self._n_tot_qubits = self._n_state_qubits + self._n_ancilla
        self._abegin = self._n_state_qubits
        self._aend = self._n_tot_qubits - 1

        self._n_classical_params = 0
        self._n_pauli_trm_measures = nruns

        # Print options banner (should done for all algorithms).
        self.print_options_banner()

        ######### QPE ########

        # add hadamard circ to split ancilla register
        self._Uqpe.add(self.get_Uhad())

        # add initial state preparation circuit
        self._Uqpe.add(self._Uprep)

        # add controll e^-iHt circuit
        self._Uqpe.add(self.get_dynamics_circ())

        # add reverse QFT
        self._Uqpe.add(self.get_qft_circuit('reverse'))

        computer = qforte.QuantumComputer(self._n_tot_qubits)
        computer.apply_circuit(self._Uqpe)

        self._n_cnot = self._Uqpe.get_num_cnots()

        if(self._fast):
            z_readouts = computer.measure_z_readouts_fast(self._abegin, self._aend, self._nruns)
        else:
            Zcirc = self.get_z_circuit()
            z_readouts = computer.measure_readouts(Zcirc, self._nruns)

        self._phases = []
        for readout in z_readouts:
            val = 0.0
            i = 1
            for z in readout:
                val += z / (2**i)
                i += 1
            self._phases.append(val)

        # find final binary string of phase readouts:
        final_readout = []
        final_readout_aves = []
        for i in range(self._n_ancilla):
            iave = 0.0
            for readout in z_readouts:
                iave += readout[i]
            iave /= nruns
            final_readout_aves.append(iave)
            if (iave > (1.0/2)):
                final_readout.append(1)
            else:
                final_readout.append(0)

        self._final_phase = 0.0
        counter = 0
        for i, z in enumerate(final_readout):
            self._final_phase += z / (2**(i+1))

        Eqpe = -2 * np.pi * self._final_phase / t
        res = stats.mode(np.asarray(self._phases))
        self._mode_phase = res.mode[0]
        self._mode_energy = -2 * np.pi * self._mode_phase / t

        print('\n           ==> QPE readout averages <==')
        print('------------------------------------------------')
        for i, ave in enumerate(final_readout_aves):
            print('  bit ', i,  ': ', ave)
        print('\n  Final bit readout: ', final_readout)

        ######### QPE ########

        # set Egs
        self._Egs = Eqpe

        # set Umaxdepth
        self._Umaxdepth = self._Uqpe

        # Print summary banner (should done for all algorithms).
        self.print_summary_banner()

        # verify that required attributes were defined
        # (should done for all algorithms).
        self.verify_run()

    def run_realistic(self):
        raise NotImplementedError('run_realistic() for QPE can be done by initializing QPE with fast==False and calling run().')

    def verify_run(self):
        self.verify_required_attributes()

    def print_options_banner(self):
        print('\n-----------------------------------------------------')
        print('       Quantum Phase Estimation Algorithm   ')
        print('-----------------------------------------------------')

        print('\n\n                 ==> QPE options <==')
        print('-----------------------------------------------------------')
        # General algorithm options.
        print('Trial reference state:                   ',  ref_string(self._ref, self._nqb))
        print('Trial state preparation method:          ',  self._trial_state_type)
        print('Trotter order (rho):                     ',  self._trotter_order)
        print('Trotter number (m):                      ',  self._trotter_number)
        print('Use fast version of algorithm:           ',  str(self._fast))
        if(self._fast):
            print('Measurement varience thresh:             ',  'NA')
        else:
            print('Measurement varience thresh:             ',  0.01)

        # Specific QPE options.
        print('Target success probability:              ',  self._success_prob)
        print('Number of precise bits for phase:        ',  self._num_precise_bits)
        print('Number of time steps:                    ',  self._n_ancilla)
        print('Evolution time (t):                      ',  self._t)
        print('Number of QPE algorithm executions:      ',  self._nruns)
        print('\n')

    def print_summary_banner(self):
        print('\n\n                        ==> QPE summary <==')
        print('---------------------------------------------------------------')
        print('Final QPE Energy:                        ',  round(self._Egs, 10))
        print('Mode QPE Energy:                         ',  round(self._mode_energy, 10))
        print('Final QPE phase:                          ', round(self._final_phase, 10))
        print('Mode QPE phase:                           ', round(self._mode_phase, 10))
        print('Number of classical parameters used:      ', self._n_classical_params)
        print('Number of CNOT gates in deepest circuit:  ', self._n_cnot)
        print('Number of Pauli term measurements:        ', self._n_pauli_trm_measures)

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

            qft_circ : QuantumCircuit
                A circuit of consecutive Hadamard gates.
        """
        Uhad = qforte.QuantumCircuit()
        for j in range(self._abegin, self._aend + 1):
            Uhad.add(qforte.gate('H', j, j))

        return Uhad

    def get_dynamics_circ(self):
        """Generates a circuit for controlled dynamics operations used in phase
        estimation. It approximates the exponentiated hermetina operator H as e^-iHt.

            Arguments
            ---------

            H : QuantumOperator
                The hermetian operaotr whos dynamics and eigenstates are of interest,
                ususally the Hamiltonian.

            trotter_num : int
                The trotter number (m) to use for the decompostion. Exponentiation
                is exact in the m --> infinity limit.

            self._abegin : int
                The index of the begin qubit.

            n_ancilla : int
                The number of anciall qubit used for the phase estimation.
                Determintes the total number of time steps.

            t : float
                The total evolution time.

            Returns
            -------

            Udyn : QuantumCircuit
                A circuit approximating controlled application of e^-iHt.
        """
        Udyn = qforte.QuantumCircuit()
        ancilla_idx = self._abegin
        total_phase = 1.0
        for n in range(self._n_ancilla):
            tn = 2 ** n
            temp_op = qforte.QuantumOperator()
            scaler_terms = []
            for h in self._qb_ham.terms():
                c, op = h
                phase = -1.0j * self._t * c #* tn
                temp_op.add(phase, op)
                gates = op.gates()
                if op.size() == 0:
                    scaler_terms.append(c * self._t)


            expn_op, phase1 = trotterize_w_cRz(temp_op,
                                               ancilla_idx,
                                               trotter_number=self._trotter_number)

            # Rotation for the scaler Hamiltonian term
            Udyn.add(qforte.gate('R', ancilla_idx, ancilla_idx,  -1.0 * np.sum(scaler_terms) * float(tn)))

            # NOTE: Approach uses 2^ancilla_idx blocks of the time evolution circuit
            for i in range(tn):
                for gate in expn_op.gates():
                    Udyn.add(gate)

            ancilla_idx += 1

        return Udyn


    def get_qft_circuit(self, direct):
        """Generates a circuit for Quantum Fourier Transformation with no swaping
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

            qft_circ : QuantumCircuit
                A circuit representing the Quantum Fourier Transform.
        """

        qft_circ = qforte.QuantumCircuit()
        lens = self._aend - self._abegin + 1
        for j in range(lens):
            qft_circ.add(qforte.gate('H', j+self._abegin, j+self._abegin))
            for k in range(2, lens+1-j):
                phase = 2.0*np.pi/(2**k)
                qft_circ.add(qforte.gate('cR', j+self._abegin, j+k-1+self._abegin, phase))

        if direct == 'forward':
            return qft_circ
        elif direct == 'reverse':
            return qft_circ.adjoint()
        else:
            raise ValueError('QFT directions can only be "forward" or "reverse"')

        return qft_circ

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

            z_circ : QuantumCircuit
                A circuit representing the the Z gates to be measured.
        """

        Z_circ = qforte.QuantumCircuit()
        for j in range(self._abegin, self._aend + 1):
            Z_circ.add(qforte.gate('Z', j, j))

        return Z_circ
