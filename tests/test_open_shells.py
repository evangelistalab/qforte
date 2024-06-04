from pytest import approx
from qforte import system_factory, UCCNPQE, ADAPTVQE, SPQE


class TestOpenShellSystems:
    def test_H5_uccsd_pqe(self):
        # This is a regression test

        Rhh = 1.5

        mol = system_factory(
            system_type="molecule",
            build_type="psi4",
            basis="sto-6g",
            mol_geometry=[
                ("H", (0, 0, -2 * Rhh)),
                ("H", (0, 0, -Rhh)),
                ("H", (0, 0, 0)),
                ("H", (0, 0, Rhh)),
                ("H", (0, 0, 2 * Rhh)),
            ],
            symmetry="d2h",
            multiplicity=2,
            charge=0,
            num_frozen_docc=0,
            num_frozen_uocc=0,
            run_mp2=0,
            run_ccsd=0,
            run_cisd=0,
            run_fci=0,
        )

        alg = UCCNPQE(mol, compact_excitations=True)
        alg.run(pool_type="SD")

        assert alg._Egs == approx(-2.4998604454039635, abs=1.0e-12)

    def test_H5_fci(self):
        Rhh = 1.5

        mol = system_factory(
            system_type="molecule",
            build_type="psi4",
            basis="sto-6g",
            mol_geometry=[
                ("H", (0, 0, -2 * Rhh)),
                ("H", (0, 0, -Rhh)),
                ("H", (0, 0, 0)),
                ("H", (0, 0, Rhh)),
                ("H", (0, 0, 2 * Rhh)),
            ],
            symmetry="d2h",
            multiplicity=2,
            charge=0,
            num_frozen_docc=0,
            num_frozen_uocc=0,
            run_mp2=0,
            run_ccsd=0,
            run_cisd=0,
            run_fci=1,
        )

        alg = UCCNPQE(mol, compact_excitations=True)
        alg.run(pool_type="SDTQ")

        assert alg._Egs == approx(mol.fci_energy, abs=1.0e-9)

    def test_H5_adaptvqe(self):
        # This is a regression test

        Rhh = 1.5

        mol = system_factory(
            system_type="molecule",
            build_type="psi4",
            basis="sto-6g",
            mol_geometry=[
                ("H", (0, 0, -2 * Rhh)),
                ("H", (0, 0, -Rhh)),
                ("H", (0, 0, 0)),
                ("H", (0, 0, Rhh)),
                ("H", (0, 0, 2 * Rhh)),
            ],
            symmetry="d2h",
            multiplicity=2,
            charge=0,
            num_frozen_docc=0,
            num_frozen_uocc=0,
            run_mp2=0,
            run_ccsd=0,
            run_cisd=0,
            run_fci=0,
        )

        alg = ADAPTVQE(mol, compact_excitations=True)
        alg.run(pool_type="GSD", avqe_thresh=0.1, adapt_maxiter=100)

        assert alg._Egs == approx(-2.4982780593834577, abs=1.0e-12)

    def test_H5_spqe(self):
        Rhh = 1.5

        mol = system_factory(
            system_type="molecule",
            build_type="psi4",
            basis="sto-6g",
            mol_geometry=[
                ("H", (0, 0, -2 * Rhh)),
                ("H", (0, 0, -Rhh)),
                ("H", (0, 0, 0)),
                ("H", (0, 0, Rhh)),
                ("H", (0, 0, 2 * Rhh)),
            ],
            symmetry="d2h",
            multiplicity=2,
            charge=0,
            num_frozen_docc=0,
            num_frozen_uocc=0,
            run_mp2=0,
            run_ccsd=0,
            run_cisd=0,
            run_fci=1,
        )

        alg = SPQE(mol, compact_excitations=True)
        alg.run()

        assert alg._Egs == approx(mol.fci_energy, abs=0.0001)

    def test_H6_cation_quartet_fci(self):
        Rhh = 1.5

        mol = system_factory(
            system_type="molecule",
            build_type="psi4",
            basis="sto-6g",
            mol_geometry=[
                ("H", (0, 0, -5 * Rhh / 2)),
                ("H", (0, 0, -3 * Rhh / 2)),
                ("H", (0, 0, -Rhh / 2)),
                ("H", (0, 0, Rhh / 2)),
                ("H", (0, 0, 3 * Rhh / 2)),
                ("H", (0, 0, 5 * Rhh / 2)),
            ],
            symmetry="d2h",
            multiplicity=4,
            charge=1,
            num_frozen_docc=0,
            num_frozen_uocc=0,
            run_mp2=0,
            run_ccsd=0,
            run_cisd=0,
            run_fci=1,
        )

        alg = UCCNPQE(mol, compact_excitations=True)
        alg.run(pool_type="SDT")

        assert alg._Egs == approx(mol.fci_energy, abs=1.0e-9)
