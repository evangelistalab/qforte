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
