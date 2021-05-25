import unittest
import numpy as np
from qforte import qforte

class ExperimentTests(unittest.TestCase):
    def test_trotterization(self):

        circ_vec = [qforte.QuantumCircuit(), qforte.build_circuit('Z_0')]
        coef_vec = [-1.0j * 0.5, -1.0j * -0.04544288414432624]

        # the operator to be exponentiated
        minus_iH = qforte.QuantumOperator()
        for i in range(len(circ_vec)):
            minus_iH.add(coef_vec[i], circ_vec[i])

        # exponentiate the operator
        Utrot, phase = qforte.trotterization.trotterize(minus_iH)

        inital_state = np.zeros(2**4, dtype=complex)
        inital_state[3]  =  np.sqrt(2/3)
        inital_state[12] = -np.sqrt(1/3)

        # initalize a quantum computer with above coeficients
        # i.e. ca|1100> + cb|0011>
        qc = qforte.QuantumComputer(4)
        qc.set_coeff_vec(inital_state)

        # apply the troterized minus_iH
        qc.apply_circuit(Utrot)
        qc.apply_constant(phase)

        qforte.smart_print(qc)

        coeffs = qc.get_coeff_vec()

        self.assertAlmostEqual(np.real(coeffs[3]),    0.6980209737879599)
        self.assertAlmostEqual(np.imag(coeffs[3]),   -0.423595782342996)
        self.assertAlmostEqual(np.real(coeffs[12]), -0.5187235657531178)
        self.assertAlmostEqual(np.imag(coeffs[12]),  0.25349397560041553)

    def test_trotterization_with_controlled_U(self):

        circ_vec = [qforte.build_circuit('Y_0 X_1'), qforte.build_circuit('X_0 Y_1')]
        coef_vec = [-1.0719145972781818j, 1.0719145972781818j]

        # the operator to be exponentiated
        minus_iH = qforte.QuantumOperator()
        for i in range(len(circ_vec)):
            minus_iH.add(coef_vec[i], circ_vec[i])

        ancilla_idx = 2

        # exponentiate the operator
        Utrot, phase = qforte.trotterization.trotterize_w_cRz(minus_iH, ancilla_idx)

        # Case 1: positive control

        # initalize a quantum computer
        qc = qforte.QuantumComputer(3)

        # build HF state
        qc.apply_gate(qforte.gate('X', 0, 0))

        # put ancilla in |1> state
        qc.apply_gate(qforte.gate('X', 2, 2))

        # apply the troterized minus_iH
        qc.apply_circuit(Utrot)

        qforte.smart_print(qc)

        coeffs = qc.get_coeff_vec()

        self.assertAlmostEqual(coeffs[5],-0.5421829373 +0.0j)
        self.assertAlmostEqual(coeffs[6],-0.840260473 +0.0j)

        # Case 2: negitive control

        # initalize a quantum computer
        qc = qforte.QuantumComputer(3)

        # build HF state
        qc.apply_gate(qforte.gate('X', 0, 0))

        # apply the troterized minus_iH
        qc.apply_circuit(Utrot)

        qforte.smart_print(qc)

        coeffs = qc.get_coeff_vec()

        self.assertAlmostEqual(coeffs[1], 1.0 +0.0j)

if __name__ == '__main__':
    unittest.main()
