import unittest
from qforte import qforte
from qforte.ucc.uccnvqe import UCCNVQE
from qforte.system.molecular_info import Molecule
from qforte.system import system_factory

import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(THIS_DIR, 'He-ccpvdz.json')

class UccTests(unittest.TestCase):
    def test_He_uccsd_exact(self):
        print('\n')
        # The FCI energy for He atom in a cc-pvdz basis
        Efci = -2.887594831090938
        # The Nuclear repulsion energy
        Enuc =  0.0

        mol_adapter = system_factory(stytem_type = 'molecule',
                                     build_type = 'external',
                                     basis='cc-pvdz',
                                     filename=data_path)

        mol_adapter.run()
        mol = mol_adapter.get_molecule()

        ref = [1,1,0,0,0,0,0,0,0,0]

        alg = UCCNVQE(mol, ref)
        alg.run(pool_type = 'SD',
                use_analytic_grad = True)

        Egs_elec = alg.get_gs_energy()
        Egs = Egs_elec + Enuc
        self.assertLess(abs(Egs-Efci), 1.0e-11)


if __name__ == '__main__':
    unittest.main()
