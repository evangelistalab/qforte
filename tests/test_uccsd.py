from pytest import approx
from qforte import system_factory, UCCNVQE, UCCNPQE

import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(THIS_DIR, 'He-ccpvdz.json')

# Note: These system are all a single atom, so we can ignore nuclear repulsion energy.

class TestUcc:
    def test_He_uccsd_vqe_exact(self):
        print('\n')
        # The FCI energy for He atom in a cc-pvdz basis
        Efci = -2.887594831090938

        mol = system_factory(system_type = 'molecule',
                                     build_type = 'external',
                                     basis='cc-pvdz',
                                     filename=data_path)


        alg = UCCNVQE(mol)
        alg.run(pool_type = 'SD',
                use_analytic_grad = True)

        Egs_elec = alg.get_gs_energy()
        assert Egs_elec == approx(Efci, abs=1.0e-10)

    def test_He_uccsd_vqe_exact_diis(self):
        print('\n')
        # The FCI energy for He atom in a cc-pvdz basis
        Efci = -2.887594831090938

        mol = system_factory(system_type = 'molecule',
                                     build_type = 'external',
                                     basis='cc-pvdz',
                                     filename=data_path)


        alg = UCCNVQE(mol)
        alg.run(pool_type = 'SD',
                use_analytic_grad = True,
                optimizer = "diis_solve")

        Egs_elec = alg.get_gs_energy()
        assert Egs_elec == approx(Efci, abs=1.0e-11)

    def test_He_uccsd_vqe_exact_psi(self):
        print('\n')
        # The FCI energy for He atom in a cc-pvdz basis
        Efci = -2.887594831090938

        mol = system_factory(system_type = 'molecule',
                                     build_type = 'psi4',
                                     basis='cc-pvdz',
                                     mol_geometry = [('He', (0, 0, 0))],
                                     symmetry = "c2v")

        alg = UCCNVQE(mol)
        alg.run(pool_type = 'SD',
                use_analytic_grad = True)

        Egs_elec = alg.get_gs_energy()
        assert Egs_elec == approx(Efci, abs=1.0e-10)

    def test_He_uccsd_vqe_frozen_virtual(self):
        print('\n')
        # The FCI energy for He atom in a cc-pvdz basis, according to Psi, one frozen virtual
        Efci = -2.881925090255593

        mol = system_factory(system_type = 'molecule',
                                     build_type = 'openfermion',
                                     basis='cc-pvdz',
                                     mol_geometry = [('He', (0, 0, 0))],
                                     virtual_indices = [8, 9])

        alg = UCCNVQE(mol)
        alg.run(pool_type = 'SD',
                use_analytic_grad = True)

        Egs_elec = alg.get_gs_energy()
        assert Egs_elec == approx(Efci, abs=1.0e-11)

    def test_He_uccsd_pqe_exact(self):
        print('\n')
        # The FCI energy for He atom in a cc-pvdz basis
        Efci = -2.887594831090938

        mol = system_factory(system_type = 'molecule',
                                     build_type = 'external',
                                     basis='cc-pvdz',
                                     filename=data_path)

        alg = UCCNPQE(mol)
        alg.run(pool_type = 'SD',
                opt_thresh = 1.0e-7)

        Egs_elec = alg.get_gs_energy()
        assert Egs_elec == approx(Efci, abs=1.0e-11)
