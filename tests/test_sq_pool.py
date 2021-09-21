from qforte import SQOpPool, SQOperator

class TestSqPool:
    def test_terms(self):
        pool = SQOpPool()
        terms = pool.terms()
        assert len(terms) == 0
        assert len(pool) == 0
        sq_op = SQOperator()
        sq_op.add_term( 1, [0, 1], [2, 3])
        sq_op.add_term(-1, [2, 3], [0, 1])
        pool.add_term(1, sq_op)
        assert len(pool) == 1
        assert len(terms) == 0

        for coefficient, operator in pool:
            assert isinstance(coefficient, complex)
            assert isinstance(operator, SQOperator)
