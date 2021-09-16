from pytest import approx
from qforte import ADAPTVQE
from qforte import system_factory

import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(THIS_DIR, 'H4-sto6g-075a.json')

class TestADAPTVQE:
    def test_H4_adapt_vqe_exact(self):
        print('\n')

        # The FCI energy for H4 at 0.75 Angstrom in a sto-6g basis
        Efci = -2.1628978832666865
        # The Nuclear repulsion energy
        Enuc =  3.057468328315556

        mol = system_factory(stytem_type = 'molecule',
                                     build_type = 'external',
                                     basis='sto-6g',
                                     filename=data_path)

        alg = ADAPTVQE(mol, print_summary_file=False)

        alg.run(adapt_maxiter=20,
                avqe_thresh=1.0e-4,
                opt_thresh=1.0e-5,
                use_analytic_grad=True,
                pool_type='SDTQ')

        Egs_elec = alg.get_gs_energy()
        # Egs = Egs_elec + Enuc
        Egs = Egs_elec
        assert Egs == approx(Efci, abs=5.0e-11)
