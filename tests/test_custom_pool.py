from pytest import approx
from qforte import SQOpPool, SQOperator, system_factory, UCCNVQE


class TestCustomPool:
    def test_vqe(self):
        print("\n")
        Efci = -1.108873060057971

        mol = system_factory(
            system_type="molecule",
            build_type="psi4",
            basis="sto-6g",
            mol_geometry=[("H", (0, 0, 0)), ("H", (0, 0, 1))],
            symmetry="d2h",
        )

        pool = SQOpPool()
        sq_op = SQOperator()
        sq_op.add_term(1, [0, 1], [2, 3])
        sq_op.add_term(-1, [2, 3], [0, 1])
        pool.add_term(1, sq_op)

        alg = UCCNVQE(mol)
        alg.run(pool_type=pool, use_analytic_grad=True)

        assert alg.get_gs_energy() == approx(Efci, abs=1.0e-10)
