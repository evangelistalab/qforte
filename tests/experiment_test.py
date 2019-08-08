import unittest
from qforte import qforte

class ExperimentTests(unittest.TestCase):
    def test_H2_experiment(self):
        print('\n')
        #the RHF H2 energy at equilibrium bond length
        E_hf = -1.1166843870661929

        #the H2 qubit hamiltonian
        circ_vec = [qforte.QuantumCircuit(),
        qforte.build_circuit('Z_0'),
        qforte.build_circuit('Z_1'),
        qforte.build_circuit('Z_2'),
        qforte.build_circuit('Z_3'),
        qforte.build_circuit('Z_0 Z_1'),
        qforte.build_circuit('Y_0 X_1 X_2 Y_3'),
        qforte.build_circuit('Y_0 Y_1 X_2 X_3'),
        qforte.build_circuit('X_0 X_1 Y_2 Y_3'),
        qforte.build_circuit('X_0 Y_1 Y_2 X_3'),
        qforte.build_circuit('Z_0 Z_2'),
        qforte.build_circuit('Z_0 Z_3'),
        qforte.build_circuit('Z_1 Z_2'),
        qforte.build_circuit('Z_1 Z_3'),
        qforte.build_circuit('Z_2 Z_3')]

        coef_vec = [-0.098863969784274,
        0.1711977489805748,
        0.1711977489805748,
        -0.222785930242875,
        -0.222785930242875,
        0.1686221915724993,
        0.0453222020577776,
        -0.045322202057777,
        -0.045322202057777,
        0.0453222020577776,
        0.1205448220329002,
        0.1658670240906778,
        0.1658670240906778,
        0.1205448220329002,
        0.1743484418396386]

        H2_qubit_hamiltonian = qforte.QuantumOperator()
        for i in range(len(circ_vec)):
            H2_qubit_hamiltonian.add_term(coef_vec[i], circ_vec[i])

        # circuit for making HF state
        circ = qforte.QuantumCircuit()
        circ.add_gate(qforte.make_gate('X', 0, 0))
        circ.add_gate(qforte.make_gate('X', 1, 1))

        TestExperiment = qforte.Experiment(4, circ, H2_qubit_hamiltonian, 1000000)
        params2 = []
        avg_energy = TestExperiment.experimental_avg(params2)
        print('Measured H2 Experimental Avg. Energy')
        print(avg_energy)
        print('H2 RHF Energy')
        print(E_hf)

        experimental_error = abs(avg_energy - E_hf)

        self.assertLess(experimental_error, 2.0e-4)

if __name__ == '__main__':
    unittest.main()
