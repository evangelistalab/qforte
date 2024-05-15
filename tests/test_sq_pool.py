# import numpy as np
from pytest import approx
from qforte import SQOpPool, SQOperator


class TestSqPool:
    def test_terms(self):
        pool = SQOpPool()
        terms = pool.terms()
        assert len(terms) == 0
        assert len(pool) == 0
        sq_op = SQOperator()
        sq_op.add_term(1, [0, 1], [2, 3])
        sq_op.add_term(-1, [2, 3], [0, 1])
        pool.add_term(1, sq_op)
        assert len(pool) == 1
        assert len(terms) == 0

        for coefficient, operator in pool:
            assert isinstance(coefficient, complex)
            assert isinstance(operator, SQOperator)

    def test_sa_sd_pool(self):
        pool = SQOpPool()
        pool.set_orb_spaces([1, 1, 0, 0, 0, 0])
        pool.fill_pool("sa_SD")

        for pool_term in pool.terms():
            coeff_norm = 0
            for sq_op_term in pool_term[1].terms():
                coeff_norm += sq_op_term[0] ** 2
            assert coeff_norm == approx(2, abs=1.0e-15)
