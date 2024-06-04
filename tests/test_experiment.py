from pytest import approx
from qforte import Circuit, gate, build_circuit, QubitOperator, Experiment


class TestExperiment:
    def test_H2_experiment(self):
        print("\n")
        # the RHF H2 energy at equilibrium bond length
        E_hf = -1.1166843870661929

        # the H2 qubit hamiltonian
        circ_vec = [
            Circuit(),
            build_circuit("Z_0"),
            build_circuit("Z_1"),
            build_circuit("Z_2"),
            build_circuit("Z_3"),
            build_circuit("Z_0 Z_1"),
            build_circuit("Y_0 X_1 X_2 Y_3"),
            build_circuit("Y_0 Y_1 X_2 X_3"),
            build_circuit("X_0 X_1 Y_2 Y_3"),
            build_circuit("X_0 Y_1 Y_2 X_3"),
            build_circuit("Z_0 Z_2"),
            build_circuit("Z_0 Z_3"),
            build_circuit("Z_1 Z_2"),
            build_circuit("Z_1 Z_3"),
            build_circuit("Z_2 Z_3"),
        ]

        coef_vec = [
            -0.098863969784274,
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
            0.1743484418396386,
        ]

        H2_qubit_hamiltonian = QubitOperator()
        for i in range(len(circ_vec)):
            H2_qubit_hamiltonian.add(coef_vec[i], circ_vec[i])

        # circuit for making HF state
        circ = Circuit()
        circ.add(gate("X", 0, 0))
        circ.add(gate("X", 1, 1))

        TestExperiment = Experiment(4, circ, H2_qubit_hamiltonian, 1000000)
        avg_energy = TestExperiment.experimental_avg()
        print("Measured H2 Experimental Avg. Energy")
        print(avg_energy)
        print("H2 RHF Energy")
        print(E_hf)

        experimental_error = abs(avg_energy - E_hf)

        assert experimental_error < 4.0e-4

    def test_H2_experiment_perfect(self):
        print("\n")
        # the RHF H2 energy at equilibrium bond length
        E_hf = -1.1166843870661929

        # the H2 qubit hamiltonian
        circ_vec = [
            Circuit(),
            build_circuit("Z_0"),
            build_circuit("Z_1"),
            build_circuit("Z_2"),
            build_circuit("Z_3"),
            build_circuit("Z_0 Z_1"),
            build_circuit("Y_0 X_1 X_2 Y_3"),
            build_circuit("Y_0 Y_1 X_2 X_3"),
            build_circuit("X_0 X_1 Y_2 Y_3"),
            build_circuit("X_0 Y_1 Y_2 X_3"),
            build_circuit("Z_0 Z_2"),
            build_circuit("Z_0 Z_3"),
            build_circuit("Z_1 Z_2"),
            build_circuit("Z_1 Z_3"),
            build_circuit("Z_2 Z_3"),
        ]

        coef_vec = [
            -0.098863969784274,
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
            0.1743484418396386,
        ]

        H2_qubit_hamiltonian = QubitOperator()
        for i in range(len(circ_vec)):
            H2_qubit_hamiltonian.add(coef_vec[i], circ_vec[i])

        # circuit for making HF state
        circ = Circuit()
        circ.add(gate("X", 0, 0))
        circ.add(gate("X", 1, 1))

        TestExperiment = Experiment(4, circ, H2_qubit_hamiltonian, 1000000)
        avg_energy = TestExperiment.perfect_experimental_avg()
        print("Perfectly Measured H2 Experimental Avg. Energy")
        print(avg_energy)
        print("H2 RHF Energy")
        print(E_hf)

        experimental_error = abs(avg_energy - E_hf)

        assert experimental_error == approx(0, abs=1.0e-14)
