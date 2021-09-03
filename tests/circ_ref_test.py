import unittest
from qforte import qforte
from qforte.ucc.uccnvqe import UCCNVQE
from qforte.qkd.srqk import SRQK
from qforte.system.molecular_info import Molecule
from qforte.system import system_factory

import os

# THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# data_path = os.path.join(THIS_DIR, 'He-ccpvdz.json')

class CircuitReferenceTest(unittest.TestCase):
    def test_srqk_with_uccsdpqe_ref(self):
        print('\nSTART test_srqk_with_uccsdpqe_ref\n')
        # The FCI energy of the equidistant H4/STO-3G chain with r = 1.5 angs,.
        Efci = -1.9961503253000235

        geom = [
        ('H', (0.0, 0.0, 0.0)),
        ('H', (0.0, 0.0, 1.5)),
        ('H', (0.0, 0.0, 3.0)),
        ('H', (0.0, 0.0, 4.5))
        ]


        mol = system_factory(system_type = 'molecule',
                             build_type = 'psi4',
                             basis='sto-3g',
                             mol_geometry = geom)


        alg = UCCNVQE(mol)
        alg.run(pool_type = 'SD')

        ref_circ = alg.build_Uvqc()

        alg2 = SRQK(mol, reference=ref_circ, state_prep_type='unitary_circ', trotter_number=150)
        alg2.run(dt=1.0, s=12)

        Egs = alg2.get_gs_energy()
        self.assertLess(abs(Egs-Efci), 5.0e-6)




if __name__ == '__main__':
    unittest.main()
