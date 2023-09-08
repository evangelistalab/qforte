from pytest import approx
from qforte import Computer, Circuit, gate
import numpy as np

class TestCircuit:
    def test_circuit(self):
        print('\n')
        num_qubits = 10

        qc1 = Computer(num_qubits)
        qc2 = Computer(num_qubits)

        prep_circ = Circuit()
        circ = Circuit()

        for i in range(num_qubits):
            prep_circ.add(gate('H',i, i))

        for i in range(num_qubits):
            prep_circ.add(gate('cR',i, i+1, 1.116 / (i+1.0)))

        for i in range(num_qubits - 1):
            circ.add(gate('cX',i, i+1))
            circ.add(gate('cX',i+1, i))
            circ.add(gate('cY',i, i+1))
            circ.add(gate('cY',i+1, i))
            circ.add(gate('cZ',i, i+1))
            circ.add(gate('cZ',i+1, i))
            circ.add(gate('cR',i, i+1, 3.14159 / (i+1.0)))
            circ.add(gate('cR',i+1, i, 2.17284 / (i+1.0)))


        qc1.apply_circuit_safe(prep_circ)
        qc2.apply_circuit_safe(prep_circ)

        qc1.apply_circuit_safe(circ)
        qc2.apply_circuit(circ)

        C1 = qc1.get_coeff_vec()
        C2 = qc2.get_coeff_vec()

        diff_vec = [ (C1[i] - C2[i])*np.conj(C1[i] - C2[i]) for i in range(len(C1))]
        diff_norm = np.sum(diff_vec)

        print('\nNorm of diff vec |C - Csafe|')
        print('-----------------------------')
        print('   ', diff_norm)
        assert diff_norm == approx(0, abs=1.0e-16)

    def test_circuit2(self):
        circ = Circuit()
        circ.add(gate('X', 0))
        circ.add(gate('Y', 1))

        # test insert_gate
        circ.insert_gate(1, gate('Z', 2))
        assert circ.size() == 3
        assert circ.gates()[1] == gate('Z', 2)
        circ.insert_gate(1, gate('X', 3))
        assert circ.size() == 4
        assert circ.gates()[1] == gate('X', 3)
        assert circ.gates()[2] == gate('Z', 2)
    
        # test remove_gate
        circ.remove_gate(2)
        assert circ.size() == 3
        assert circ.gates()[1] == gate('X', 3)
        assert circ.gates()[2] == gate('Y', 1)

        # test swap_gates
        circ.swap_gates(1, 2)
        assert circ.size() == 3
        assert circ.gates()[1] == gate('Y', 1)
        assert circ.gates()[2] == gate('X', 3)

        # test insert_circuit
        circ2 = Circuit()
        circ2.add(gate('Z', 4))
        circ2.add(gate('X', 5))
        circ.insert_circuit(1, circ2)
        assert circ.size() == 5
        assert circ.gates()[0] == gate('X', 0)
        assert circ.gates()[1] == gate('Z', 4)
        assert circ.gates()[2] == gate('X', 5)
        assert circ.gates()[3] == gate('Y', 1)
        assert circ.gates()[4] == gate('X', 3)

        # test remove_gates
        circ.remove_gates(1, 3)
        assert circ.size() == 3
        assert circ.gates()[0] == gate('X', 0)
        assert circ.gates()[1] == gate('Y', 1)
        assert circ.gates()[2] == gate('X', 3)

        # test replace_gate
        circ.replace_gate(1, gate('Z', 4))
        assert circ.size() == 3
        assert circ.gates()[0] == gate('X', 0)
        assert circ.gates()[1] == gate('Z', 4)
        assert circ.gates()[2] == gate('X', 3)

        # test gates
        assert circ.gates() == [gate('X', 0), gate('Z', 4), gate('X', 3)]

        # test gate
        assert circ.gate(0) == gate('X', 0)
        assert circ.gate(1) == gate('Z', 4)
        assert circ.gate(2) == gate('X', 3)
        