import unittest
import numpy as np
from qforte import qforte
from qforte.qpea.qpe_helpers import *

class QPETests(unittest.TestCase):
    def test_H4(self):
        print('\n'),
        # The FCI energy for H2 at 1.5 Angstrom in a sto-3g basis
        E_fci = -0.9981493534

        coef_vec = [-0.4917857774144603,
                    0.09345649662931771,
                    0.09345649662931771,
                    -0.0356448161226769,
                    -0.0356448161226769,
                    0.1381758457453024,
                    0.05738398402634884,
                    -0.0573839840263488,
                    -0.0573839840263488,
                    0.05738398402634884,
                    0.08253705485911705,
                    0.13992103888546592,
                    0.13992103888546592,
                    0.08253705485911705,
                    0.1458551902800438]

        circ_vec = [
        qforte.QuantumCircuit( ),
        qforte.build_circuit( 'Z_0' ),
        qforte.build_circuit( 'Z_1' ),
        qforte.build_circuit( 'Z_2' ),
        qforte.build_circuit( 'Z_3' ),
        qforte.build_circuit( 'Z_0   Z_1' ),
        qforte.build_circuit( 'Y_0   X_1   X_2   Y_3' ),
        qforte.build_circuit( 'X_0   X_1   Y_2   Y_3' ),
        qforte.build_circuit( 'Y_0   Y_1   X_2   X_3' ),
        qforte.build_circuit( 'X_0   Y_1   Y_2   X_3' ),
        qforte.build_circuit( 'Z_0   Z_2' ),
        qforte.build_circuit( 'Z_0   Z_3' ),
        qforte.build_circuit( 'Z_1   Z_2' ),
        qforte.build_circuit( 'Z_1   Z_3' ),
        qforte.build_circuit( 'Z_2   Z_3' )]

        H2_qubit_hamiltonian = qforte.QuantumOperator()
        for i in range(len(circ_vec)):
            H2_qubit_hamiltonian.add_term(coef_vec[i], circ_vec[i])

        ref = [1,1,0,0]

        print('\nBegin QPE test for H2')
        print('----------------------')

        n_state_qubits = len(ref)
        n_ancilla = 10
        n_tot_qubits = n_state_qubits + n_ancilla
        trotter_number = 2
        t = 0.4
        nruns = 100

        abegin = n_state_qubits
        aend = n_tot_qubits - 1

        # build hadamard circ
        Uhad = get_Uhad(abegin, aend)

        # build preparation circuit
        Uprep = get_Uprep(ref, 'single_reference')

        # build controll e^-iHt circuit
        Udyn = get_dynamics_circ(H2_qubit_hamiltonian,
                                 trotter_number,
                                 abegin,
                                 n_ancilla,
                                 t=t)

        # build reverse QFT
        revQFTcirc = qft_circuit(abegin, aend, 'reverse')

        # build QPEcirc
        QPEcirc = qforte.QuantumCircuit()
        QPEcirc.add_circuit(Uprep)
        QPEcirc.add_circuit(Uhad)
        QPEcirc.add_circuit(Udyn)
        QPEcirc.add_circuit(revQFTcirc)

        computer = qforte.QuantumComputer(n_tot_qubits)
        computer.apply_circuit(QPEcirc)

        z_readouts = computer.measure_z_readouts_fast(abegin, aend, nruns)

        final_energy = 0.0
        phases = []
        for readout in z_readouts:
            val = 0.0
            i = 1
            for z in readout:
                val += z / (2**i)
                i += 1
            phases.append(val)

        # find final binary string:
        final_readout = []
        final_readout_aves = []
        for i in range(n_ancilla):
            iave = 0.0
            for readout in z_readouts:
                iave += readout[i]
            iave /= nruns
            final_readout_aves.append(iave)
            if (iave > (1.0/2)):
                final_readout.append(1)
            else:
                final_readout.append(0)

        print('\n           ==> QPE readout averages <==')
        print('------------------------------------------------')
        for i, ave in enumerate(final_readout_aves):
            print('  bit ', i,  ': ', ave)
        print('\n  Final bit readout: ', final_readout)

        final_phase = 0.0
        counter = 0
        for i, z in enumerate(final_readout):
                final_phase += z / (2**(i+1))

        final_energy = -2 * np.pi * final_phase / t
        print('Eqpe: ', final_energy)
        print('Efci: ', E_fci)
        self.assertLess(np.abs(final_energy-E_fci), 1.0e-2)


if __name__ == '__main__':
    unittest.main()
