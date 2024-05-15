from pytest import approx
from qforte import system_factory, UCCNVQE, UCCNPQE

import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(THIS_DIR, "He-ccpvdz.json")

# Note: These system are all a single atom, so we can ignore nuclear repulsion energy.


class TestUcc:
    def test_He_uccsd_vqe_exact(self):
        print("\n")
        # The FCI energy for He atom in a cc-pvdz basis
        Efci = -2.887594831090938

        mol = system_factory(
            system_type="molecule",
            build_type="external",
            basis="cc-pvdz",
            filename=data_path,
        )

        alg = UCCNVQE(mol)
        alg.run(pool_type="SD", use_analytic_grad=True)

        Egs_elec = alg.get_gs_energy()
        assert Egs_elec == approx(Efci, abs=1.0e-10)

    def test_He_uccsd_vqe_exact_jacobi(self):
        print("\n")
        # The FCI energy for He atom in a cc-pvdz basis
        Efci = -2.887594831090938

        mol = system_factory(
            system_type="molecule",
            build_type="external",
            basis="cc-pvdz",
            filename=data_path,
        )

        alg = UCCNVQE(mol)
        alg.run(pool_type="SD", use_analytic_grad=True, optimizer="jacobi")

        Egs_elec = alg.get_gs_energy()
        assert Egs_elec == approx(Efci, abs=1.0e-11)

    def test_He_uccsd_vqe_exact_psi(self):
        print("\n")
        # The FCI energy for He atom in a cc-pvdz basis
        Efci = -2.887594831090938

        mol = system_factory(
            system_type="molecule",
            build_type="psi4",
            basis="cc-pvdz",
            mol_geometry=[("He", (0, 0, 0))],
            symmetry="c2v",
        )

        alg = UCCNVQE(mol)
        alg.run(pool_type="SD", use_analytic_grad=True)

        Egs_elec = alg.get_gs_energy()
        assert Egs_elec == approx(Efci, abs=1.0e-10)

    def test_He_uccsd_pqe_exact(self):
        print("\n")
        # The FCI energy for He atom in a cc-pvdz basis
        Efci = -2.887594831090938

        mol = system_factory(
            system_type="molecule",
            build_type="external",
            basis="cc-pvdz",
            filename=data_path,
        )

        alg = UCCNPQE(mol)
        alg.run(pool_type="SD", opt_thresh=1.0e-7)

        Egs_elec = alg.get_gs_energy()
        assert Egs_elec == approx(Efci, abs=1.0e-11)

    def test_uccsd_scipy_solver(self):
        # In this test, we confirm that the UCCNPQE algorithm produces
        # identical results when using the Jacobi and BFGS solvers

        Rhh = 2

        mol = system_factory(
            system_type="molecule",
            build_type="psi4",
            basis="sto-6g",
            mol_geometry=[
                ("H", (0, -Rhh / 2, -Rhh / 2)),
                ("H", (0, -Rhh / 2, Rhh / 2)),
                ("H", (0, Rhh / 2, -Rhh / 2)),
                ("H", (0, Rhh / 2, Rhh / 2)),
            ],
            symmetry="d2h",
            multiplicity=1,  # Only singlets will work with QForte
            charge=0,
            num_frozen_docc=0,
            num_frozen_uocc=0,
            run_mp2=1,
            run_ccsd=0,
            run_cisd=0,
            run_fci=1,
        )

        jacobi = UCCNPQE(
            mol, compact_excitations=True, qubit_excitations=False, diis_max_dim=8
        )
        jacobi.run(optimizer="jacobi", pool_type="SD")

        bfgs = UCCNPQE(mol, compact_excitations=True, qubit_excitations=False)
        bfgs.run(optimizer="BFGS", pool_type="SD")

        assert jacobi.get_gs_energy() == approx(bfgs.get_gs_energy(), abs=1.0e-8)
