import unittest
from qforte import qforte
from qforte.ucc.spqe import SPQE
from qforte.system.molecular_info import Molecule
from qforte.system import system_factory

import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(THIS_DIR, 'H4-sto6g-075a.json')

class SPQETests(unittest.TestCase):
    def test_H4_spqe_exact(self):
        print('\n')

        # The FCI energy for H4 at 0.75 Angstrom in a sto-6g basis
        Efci = -2.1628978832666865
        # The Nuclear repulsion energy
        Enuc =  3.057468328315556

        mol = system_factory(stytem_type = 'molecule',
                                     build_type = 'external',
                                     basis='sto-6g',
                                     filename=data_path)

        Hnonzero = qforte.QuantumOperator()
        for term in mol._hamiltonian.terms():
            if abs(term[0]) > 1.0e-14:
                Hnonzero.add_term(term[0], term[1])
        mol._hamiltonian = Hnonzero
        qforte.smart_print(Hnonzero)

        ref = [1, 1, 1, 1, 0, 0, 0, 0]

        alg = SPQE(mol, ref, print_summary_file=False)
        alg.run(spqe_maxiter=20,
                spqe_thresh=1.0e-4,
                res_vec_thresh=1.0e-5,
                dt = 0.0001)

        Egs_elec = alg.get_gs_energy()
        # Egs = Egs_elec + Enuc
        Egs = Egs_elec
        self.assertLess(abs(Egs-Efci), 5.0e-11)


if __name__ == '__main__':
    unittest.main()
