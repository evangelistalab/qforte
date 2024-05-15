from pytest import approx
from qforte import (
    Computer,
    Circuit,
    gate,
    inner_product,
    prepare_computer_from_circuit,
    prepare_computer_from_gates,
    add_gate_to_computer,
)
import numpy as np


class TestComputer:
    def test_computer(self):
        print("\n")
        num_qubits = 8

        qc1 = Computer(num_qubits)
        circ = Circuit()
        circ.add(gate("H", 0))
        circ.add(gate("H", 1))
        circ.add(gate("H", 2))
        circ.add(gate("X", 3))
        circ.add(gate("Y", 4))
        circ.add(gate("Z", 5))

        qc1.apply_circuit(circ)

        # test copy-constructor
        qc2 = Computer(qc1)

        C1 = qc1.get_coeff_vec()
        C2 = qc2.get_coeff_vec()

        diff_vec = [(x - y) * np.conj(x - y) for x, y in zip(C1, C2)]
        diff_norm = np.sum(diff_vec)

        print("\nNorm of diff vec |C - Csafe|")
        print("-----------------------------")
        print("   ", diff_norm)
        assert diff_norm == approx(0, abs=1.0e-16)

        # test null_state
        qc1.null_state()
        C1 = np.array(qc1.get_coeff_vec())
        assert np.linalg.norm(C1) == approx(0, abs=1.0e-12)

        # test reset
        qc1.reset()
        C1 = np.array(qc1.get_coeff_vec())
        assert C1[0] == approx(1, abs=1.0e-12)
        assert np.linalg.norm(C1[1:]) == approx(0, abs=1.0e-12)

        # test set_coeff_vec_from_numpy
        qc1.set_coeff_vec_from_numpy(np.array(qc2.get_coeff_vec()))
        assert inner_product(qc1, qc2) == approx(1, abs=1.0e-12)

        # test copy-constructor and equality
        qc2 = Computer(qc1)
        assert qc1 == qc2
        qc2.apply_gate(gate("H", 0))
        assert qc1 != qc2

    def test_computer_prepare(self):
        # test preparing a state
        num_qubits = 2

        # test the add_gate_to_computer function
        qc1 = Computer(num_qubits)
        qc1 = add_gate_to_computer(gate("H", 0), qc1)
        qc1 = add_gate_to_computer(gate("H", 1), qc1)
        C1 = np.array(qc1.get_coeff_vec())
        # verify that the state is |00> + |01> + |10> + |11>
        assert C1 == approx([0.5, 0.5, 0.5, 0.5], abs=1.0e-12)

        # test the prepare_computer_from_circuit function
        circ = Circuit()
        circ.add(gate("H", 0))
        circ.add(gate("H", 1))
        qc2 = prepare_computer_from_circuit(num_qubits, circ)
        assert qc2 == qc1

        # test the prepare_computer_from_gates function
        qc3 = prepare_computer_from_gates(num_qubits, [gate("H", 0), gate("H", 1)])
        assert qc1 == qc3
