from qforte import SQOperator


class TestSQOp:
    def test_mult_coeffs(self):
        # In this test, we confirm that the method `mult_coeffs` works
        # to multiply each coefficient with one identical number

        sq_op = SQOperator()

        sq_op.add_term(1, [0, 1], [2, 3])
        sq_op.add_term(-1, [2, 3], [0, 1])
        sq_op.mult_coeffs(0.5 + 0.5j)

        terms = sq_op.terms()

        assert len(terms) == 2
        assert terms[0][1] == [0, 1]
        assert terms[0][2] == [2, 3]

        assert terms[0][0] == 0.5 + 0.5j
        assert terms[1][0] == -0.5 - 0.5j
