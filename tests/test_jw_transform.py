from qforte import build_circuit, SQOperator, QubitOperator

class TestJordanWigner:
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
        build_circuit('X_1 Y_2 X_3 Y_4'),
        build_circuit('X_1 Y_2 X_3 X_4'),
        build_circuit('X_1 Y_2 Y_3 X_4'),
        build_circuit('X_1 X_2 X_3 Y_4'),
        build_circuit('X_1 X_2 X_3 X_4'),
        build_circuit('X_1 X_2 Y_3 X_4'),
        build_circuit('Y_2 X_3 X_4 Z_5 X_6'),
        build_circuit('Y_1 Y_2 Y_3 Y_4'),
        build_circuit('Y_1 X_2 X_3 Y_4'),
        build_circuit('Y_1 X_2 X_3 X_4'),
        build_circuit('Y_1 Y_2 X_3 X_4'),
        build_circuit('X_2 X_3 X_4 Z_5 X_6'),
        build_circuit('Y_1 X_2 Y_3 Y_4'),
        build_circuit('X_2 Y_3 Y_4 Z_5 Y_6'),
        build_circuit('X_2 Y_3 X_4 Z_5 X_6'),
        build_circuit('X_2 Y_3 Y_4 Z_5 X_6'),
        build_circuit('X_2 X_3 Y_4 Z_5 Y_6'),
        build_circuit('Y_2 Y_3 X_4 Z_5 X_6'),
        build_circuit('X_2 X_3 Y_4 Z_5 X_6'),
        build_circuit('Y_1 X_2 Y_3 X_4'),
        build_circuit('Y_2 Y_3 X_4 Z_5 Y_6'),
        build_circuit('Y_2 X_3 X_4 Z_5 Y_6'),
        build_circuit('X_2 X_3 X_4 Z_5 Y_6'),
        build_circuit('X_2 Y_3 X_4 Z_5 Y_6'),
        build_circuit('Y_1 Y_2 Y_3 X_4'),
        build_circuit('Y_1 Y_2 X_3 Y_4'),
        build_circuit('Y_2 Y_3 Y_4 Z_5 X_6'),
        build_circuit('X_1 X_2 Y_3 Y_4'),
        build_circuit('Y_2 Y_3 Y_4 Z_5 Y_6'),
        build_circuit('Y_2 X_3 Y_4 Z_5 X_6'),
        build_circuit('X_1 Y_2 Y_3 Y_4'),
        build_circuit('Y_2 X_3 Y_4 Z_5 Y_6')
        ]

        correct_op = QubitOperator()
        for coeff, circ in zip(coeff_vec, circ_vec):
            correct_op.add(coeff, circ)

        # Test qforte construction
        a1 = SQOperator()
        a1.add(1.00, [ 2, 3], [4, 6] )
        a1.add(+1.50, [ 1, 2], [3, 4])
        print(a1.str(), '\n')
        aop = a1.jw_transform()
        print(aop.str(), '\n')
        assert aop.check_op_equivalence(correct_op, True) is True

        #TODO: add more jw test cases

    def test_jw2(self):
        coeff_vec = [
            -0.25j,
            -0.25,
             0.25,
            -0.25j,
            -0.125j,
             0.125,
            -0.125,
            -0.125j,
            -0.125,
            -0.125j,
             0.125j,
            -0.125
        ]

        circ_vec = [build_circuit(x) for x in [
            "Y_2 X_3",
            "Y_2 Y_3",
            "X_2 X_3",
            "X_2 Y_3",
            "Z_0 X_1 Y_3 X_4",
            "Z_0 X_1 Y_3 Y_4",
            "Z_0 X_1 X_3 X_4",
            "Z_0 X_1 X_3 Y_4",
            "Z_0 Y_1 Y_3 X_4",
            "Z_0 Y_1 Y_3 Y_4",
            "Z_0 Y_1 X_3 X_4",
            "Z_0 Y_1 X_3 Y_4"
            ]]
        correct_op = QubitOperator()
        for coeff, circ in zip(coeff_vec, circ_vec):
            correct_op.add(coeff, circ)
        a = SQOperator()
        a.add(1.00, [2, 3], [])
        a.add(1.00, [1], [3, 4])
        aop = a.jw_transform()
        assert aop.check_op_equivalence(correct_op, True) is True
