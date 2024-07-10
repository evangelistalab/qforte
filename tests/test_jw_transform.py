from qforte import build_circuit, SQOperator, QubitOperator, Computer


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
            -0.062500j,
        ]

        circ_vec = [
            build_circuit("X_1 Y_2 X_3 Y_4"),
            build_circuit("X_1 Y_2 X_3 X_4"),
            build_circuit("X_1 Y_2 Y_3 X_4"),
            build_circuit("X_1 X_2 X_3 Y_4"),
            build_circuit("X_1 X_2 X_3 X_4"),
            build_circuit("X_1 X_2 Y_3 X_4"),
            build_circuit("Y_2 X_3 X_4 Z_5 X_6"),
            build_circuit("Y_1 Y_2 Y_3 Y_4"),
            build_circuit("Y_1 X_2 X_3 Y_4"),
            build_circuit("Y_1 X_2 X_3 X_4"),
            build_circuit("Y_1 Y_2 X_3 X_4"),
            build_circuit("X_2 X_3 X_4 Z_5 X_6"),
            build_circuit("Y_1 X_2 Y_3 Y_4"),
            build_circuit("X_2 Y_3 Y_4 Z_5 Y_6"),
            build_circuit("X_2 Y_3 X_4 Z_5 X_6"),
            build_circuit("X_2 Y_3 Y_4 Z_5 X_6"),
            build_circuit("X_2 X_3 Y_4 Z_5 Y_6"),
            build_circuit("Y_2 Y_3 X_4 Z_5 X_6"),
            build_circuit("X_2 X_3 Y_4 Z_5 X_6"),
            build_circuit("Y_1 X_2 Y_3 X_4"),
            build_circuit("Y_2 Y_3 X_4 Z_5 Y_6"),
            build_circuit("Y_2 X_3 X_4 Z_5 Y_6"),
            build_circuit("X_2 X_3 X_4 Z_5 Y_6"),
            build_circuit("X_2 Y_3 X_4 Z_5 Y_6"),
            build_circuit("Y_1 Y_2 Y_3 X_4"),
            build_circuit("Y_1 Y_2 X_3 Y_4"),
            build_circuit("Y_2 Y_3 Y_4 Z_5 X_6"),
            build_circuit("X_1 X_2 Y_3 Y_4"),
            build_circuit("Y_2 Y_3 Y_4 Z_5 Y_6"),
            build_circuit("Y_2 X_3 Y_4 Z_5 X_6"),
            build_circuit("X_1 Y_2 Y_3 Y_4"),
            build_circuit("Y_2 X_3 Y_4 Z_5 Y_6"),
        ]

        correct_op = QubitOperator()
        for coeff, circ in zip(coeff_vec, circ_vec):
            correct_op.add(coeff, circ)

        # Test qforte construction
        a1 = SQOperator()
        a1.add(1.00, [2, 3], [4, 6])
        a1.add(+1.50, [1, 2], [3, 4])
        print(a1.str(), "\n")
        aop = a1.jw_transform()
        print(aop.str(), "\n")
        assert aop.check_op_equivalence(correct_op, True) is True

        # TODO: add more jw test cases

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
            -0.125,
        ]

        circ_vec = [
            build_circuit(x)
            for x in [
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
                "Z_0 Y_1 X_3 Y_4",
            ]
        ]
        correct_op = QubitOperator()
        for coeff, circ in zip(coeff_vec, circ_vec):
            correct_op.add(coeff, circ)
        a = SQOperator()
        a.add(1.00, [2, 3], [])
        a.add(1.00, [1], [3, 4])
        aop = a.jw_transform()
        assert aop.check_op_equivalence(correct_op, True) is True

        def test_jw_qubit(self):
            # In this test we check the JW transform for fermionic and qubit excitations.
            # We test the action of the a_3, a^3, Q_3, Q^3 on the |1110> and |1111>
            # qubit basis states
            fermion_create_3 = qf.SQOperator()
            fermion_annihilate_3 = qf.SQOperator()
            fermion_create_3.add(1, [3], [])
            fermion_annihilate_3.add(1, [], [3])
            expected = [
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, (-1 + 0j)],
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, (1 + 0j)],
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, (-1 + 0j), 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, (1 + 0j), 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
            ]
            results = []
            for sq_operator in [fermion_create_3, fermion_annihilate_3]:
                for n_occupied in [3, 4]:
                    for qubit_excitations in [False, True]:
                        # create desired qubit state
                        comp = Computer(4)
                        for occupied in range(n_occupied):
                            comp.apply_gate(qf.gate("X", occupied))
                        # transform second-quanitzed operators using JW
                        q_operator = sq_operator.jw_transform(qubit_excitations)
                        comp.apply_operator(q_operator)
                        results.append(comp.get_coeff_vec())
            assert results == expected

        def test_jw_qubit_2(self):
            # In this test we confirm that the there exists a 1-to-1 mapping between
            # second-quantized fermionic operators and qubit excitation operators.
            # The uniqueness is guaranteed by the normal ordering of second-quantized
            # operators
            sq_op1 = SQOperator()
            sq_op1.add(1, [1, 0], [])
            q_op1 = sq_op1.jw_transform(True)
            sq_op2 = SQOperator()
            sq_op2.add(-1, [0, 1], [])
            q_op2 = sq_op2.jw_transform(True)
            assert q_op1 == q_op2
