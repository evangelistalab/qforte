import unittest
from qforte import qforte

class IOTests(unittest.TestCase):
    def test_io_simplified(self):
        #test direct expectation value measurement
        trial_state = qforte.QuantumComputer(4)

        trial_circ = qforte.build_circuit('H_0 H_1 H_2 H_3 cX_0_1')

        # use circuit to prepare trial state
        trial_state.apply_circuit(trial_circ)

        #build the quantum operator for [a1^ a2]
        a1_dag_a2 = qforte.build_operator('0.0-0.25j, X_2 Y_1; 0.25, Y_2 Y_1; \
        0.25, X_2 X_1; 0.0+0.25j, Y_2 X_1')

        #get direct expectatoin value
        exp = trial_state.direct_op_exp_val(a1_dag_a2)
        self.assertAlmostEqual(exp, 0.2499999999999999 + 0.0j)


if __name__ == '__main__':
    unittest.main()
