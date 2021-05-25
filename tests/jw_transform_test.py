import unittest
from qforte import qforte

class JordanWignerTests(unittest.TestCase):
    def test_jw1(self):
        """
        test JW transform of 1.0(2^ 3^ 4 6) + 1.5(1^ 2^ 3 4)
        """

        coeff_vec = [
        -0.093750,
        +0.093750j,
        -0.093750,
        -0.093750j,
        -0.093750,
        -0.093750j,
        +0.062500j,
        -0.093750,
        -0.093750,
        +0.093750j,
        +0.093750,
        -0.062500,
        -0.093750j,
        -0.062500j,
        +0.062500j,
        -0.062500,
        +0.062500,
        +0.062500,
        -0.062500j,
        -0.093750,
        +0.062500j,
        -0.062500,
        -0.062500j,
        -0.062500,
        +0.093750j,
        +0.093750j,
        +0.062500j,
        +0.093750,
        -0.062500,
        -0.062500,
        -0.093750j,
        -0.062500j
        ]

        circ_vec = [
        qforte.build_circuit('X_1 Y_2 X_3 Y_4'),
        qforte.build_circuit('X_1 Y_2 X_3 X_4'),
        qforte.build_circuit('X_1 Y_2 Y_3 X_4'),
        qforte.build_circuit('X_1 X_2 X_3 Y_4'),
        qforte.build_circuit('X_1 X_2 X_3 X_4'),
        qforte.build_circuit('X_1 X_2 Y_3 X_4'),
        qforte.build_circuit('Y_2 X_3 X_4 Z_5 X_6'),
        qforte.build_circuit('Y_1 Y_2 Y_3 Y_4'),
        qforte.build_circuit('Y_1 X_2 X_3 Y_4'),
        qforte.build_circuit('Y_1 X_2 X_3 X_4'),
        qforte.build_circuit('Y_1 Y_2 X_3 X_4'),
        qforte.build_circuit('X_2 X_3 X_4 Z_5 X_6'),
        qforte.build_circuit('Y_1 X_2 Y_3 Y_4'),
        qforte.build_circuit('X_2 Y_3 Y_4 Z_5 Y_6'),
        qforte.build_circuit('X_2 Y_3 X_4 Z_5 X_6'),
        qforte.build_circuit('X_2 Y_3 Y_4 Z_5 X_6'),
        qforte.build_circuit('X_2 X_3 Y_4 Z_5 Y_6'),
        qforte.build_circuit('Y_2 Y_3 X_4 Z_5 X_6'),
        qforte.build_circuit('X_2 X_3 Y_4 Z_5 X_6'),
        qforte.build_circuit('Y_1 X_2 Y_3 X_4'),
        qforte.build_circuit('Y_2 Y_3 X_4 Z_5 Y_6'),
        qforte.build_circuit('Y_2 X_3 X_4 Z_5 Y_6'),
        qforte.build_circuit('X_2 X_3 X_4 Z_5 Y_6'),
        qforte.build_circuit('X_2 Y_3 X_4 Z_5 Y_6'),
        qforte.build_circuit('Y_1 Y_2 Y_3 X_4'),
        qforte.build_circuit('Y_1 Y_2 X_3 Y_4'),
        qforte.build_circuit('Y_2 Y_3 Y_4 Z_5 X_6'),
        qforte.build_circuit('X_1 X_2 Y_3 Y_4'),
        qforte.build_circuit('Y_2 Y_3 Y_4 Z_5 Y_6'),
        qforte.build_circuit('Y_2 X_3 Y_4 Z_5 X_6'),
        qforte.build_circuit('X_1 Y_2 Y_3 Y_4'),
        qforte.build_circuit('Y_2 X_3 Y_4 Z_5 Y_6')
        ]

        correct_op = qforte.QuantumOperator()
        for i in range(len(circ_vec)):
            correct_op.add(coeff_vec[i], circ_vec[i])

        # Test qforte construction
        a1 = qforte.SQOperator()
        a1.add(1.00, [ 2, 3, 4, 6] )
        a1.add(+1.50, [ 1, 2, 3, 4])
        print(a1.str(), '\n')
        aop = a1.jw_transform()
        print(aop.str(), '\n')
        self.assertTrue(aop.check_op_equivalence(correct_op, True))

        #TODO: add more jw test cases

if __name__ == '__main__':
    unittest.main()
