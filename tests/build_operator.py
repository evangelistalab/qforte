import unittest
import qforte
from openfermion.ops import QubitOperator
#from openfermion.ops import QubitOperator

class BuilderTests(unittest.TestCase):
    def test_build_from_openfermion(self):
        print('\n')
        trial_state = qforte.Computer(4)

        trial_prep = [None]*5
        trial_prep[0] = qforte.gate('H',0,0)
        trial_prep[1] = qforte.gate('H',1,1)
        trial_prep[2] = qforte.gate('H',2,2)
        trial_prep[3] = qforte.gate('H',3,3)
        trial_prep[4] = qforte.gate('cX',0,1)

        trial_circ = qforte.QuantumCircuit()

        #prepare the circuit
        for gate in trial_prep:
            trial_circ.add(gate)

        # use circuit to prepare trial state
        trial_state.apply_circuit(trial_circ)

        test_operator = QubitOperator('X2 Y1', 0.0-0.25j)
        test_operator += QubitOperator('Y2 Y1', 0.25)
        test_operator += QubitOperator('X2 X1', 0.25)
        test_operator += QubitOperator('Y2 X1', 0.0+0.25j)
        print(test_operator)

        qforte_operator = qforte.build_from_openfermion(test_operator)

        qforte.smart_print(qforte_operator)

        exp = trial_state.direct_op_exp_val(qforte_operator)
        print(exp)
        self.assertAlmostEqual(exp, 0.2499999999999999 + 0.0j)

if __name__ == '__main__':
    unittest.main()
