import unittest
from qforte import qforte

class ExperimentTests(unittest.TestCase):
    def test_trotterization(self):

        circ_vec = [qforte.build_circuit('Y_0 X_1'), qforte.build_circuit('X_0 Y_1')]
        coef_vec = [-1.0719145972781818+0j, 1.0719145972781818+0j]

        # the operator to be exponentiated
        generator = qforte.QuantumOperator()
        for i in range(len(circ_vec)):
            generator.add_term(coef_vec[i], circ_vec[i])

        # exponentiate the operator
        troterized_gen = qforte.trotterization.trotterize(generator)

        # initalize a quantum computer
        qc = qforte.QuantumComputer(2)

        # build HF state
        qc.apply_gate(qforte.make_gate('X', 0, 0))

        # apply the troterized generator
        qc.apply_circuit(troterized_gen)

        qforte.smart_print(qc)

        coeffs = qc.get_coeff_vec()

        self.assertAlmostEqual(coeffs[1],-0.5421829373 +0.0j)
        self.assertAlmostEqual(coeffs[2], 0.840260473 +0.0j)

if __name__ == '__main__':
    unittest.main()
