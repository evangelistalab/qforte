import unittest
import qforte as qf
from qforte.ucc.uccnvqe import UCCNVQE
from qforte.system import system_factory

class CustomPoolTests(unittest.TestCase):

    def test_vqe(self):
        print('\n')
        Efci = -1.108873060057971

        mol = system_factory(system_type = 'molecule',
                                     build_type = 'psi4',
                                     basis='sto-6g',
                                     mol_geometry = [('H', (0, 0, 0)),
                                                     ('H', (0, 0, 1))],
                                     symmetry = "c2v")

        pool = qf.SQOpPool()
        sq_op = qf.SQOperator()
        sq_op.add_term( 1, [0, 1], [2, 3]) 
        sq_op.add_term(-1, [2, 3], [0, 1]) 
        pool.add_term(1, sq_op)

        alg = UCCNVQE(mol)
        alg.run(pool_type = pool,
                use_analytic_grad = True)

        self.assertAlmostEqual(alg.get_gs_energy(), Efci, 10)

if __name__ == '__main__':
    unittest.main()

