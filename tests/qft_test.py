from qforte import *
import unittest

class QFTTests(unittest.TestCase):
    def test_qft(self):
        trial_state = Computer(4)
        trial_circ = build_circuit('X_0 X_1')
        trial_state.apply_circuit(trial_circ)

        # verify direct transformation
        qft(trial_state, 0, 3)

        a1_dag_a2 = qforte.build_operator('1.0, Z_0')
        exp = trial_state.direct_op_exp_val(a1_dag_a2)
        self.assertAlmostEqual(exp, 0.0 + 0.0j)

        # test unitarity
        qft(trial_state, 0, 2)
        rev_qft(trial_state, 0, 2)

        a1_dag_a2 = qforte.build_operator('1.0, Z_0')
        exp = trial_state.direct_op_exp_val(a1_dag_a2)
        self.assertAlmostEqual(exp, 0.0 + 0.0j)

        # test reverse transformation
        qft(trial_state, 0, 3)

        a1_dag_a2 = qforte.build_operator('1.0, Z_0')
        exp = trial_state.direct_op_exp_val(a1_dag_a2)
        self.assertAlmostEqual(exp, 1.0 + 0.0j)

if __name__ == '__main__':
    unittest.main()
