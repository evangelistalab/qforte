from qforte import *
import unittest

class QFTTests(unittest.TestCase):
    def test_qft(self):
        trial_state = QuantumComputer(4)
        trial_circ = build_circuit('X_1')
        trial_state.apply_circuit(trial_circ)

        # verify transformation
        qft(trial_state, 4)

        # test unitarity
        qft(trial_state, 3)
        rev_qft(trial_state, 3)

        a1_dag_a2 = qforte.build_operator('0.0-0.25j, X_2 Y_1; 0.25, Y_2 Y_1; \
        0.25, X_2 X_1; 0.0+0.25j, Y_2 X_1')

        exp = trial_state.direct_op_exp_val(a1_dag_a2)
        self.assertAlmostEqual(exp, -0.2499999999999999 + 0.0j)
        
if __name__ == '__main__':
    unittest.main()

