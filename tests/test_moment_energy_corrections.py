from pytest import approx, mark
from qforte import system_factory, UCCNPQE


class TestNonIterativeEnergyCorrections:
    @mark.skip(reason="ambiguous test case")
    def test_N2_uccsd_pqe(self):
        # This is a regression test

        mol = system_factory(
            system_type="molecule",
            build_type="psi4",
            basis="sto-6g",
            mol_geometry=[("N", (0, 0, 0)), ("N", (0, 0, 3))],
            symmetry="d2h",
            multiplicity=1,  # Only singlets will work with QForte
            charge=0,
            num_frozen_docc=4,
            num_frozen_uocc=0,
            run_mp2=0,
            run_ccsd=0,
            run_cisd=0,
            run_fci=0,
        )

        alg = UCCNPQE(mol, compact_excitations=True, diis_max_dim=5, max_moment_rank=4)
        alg.run(pool_type="SD", opt_maxiter=60)

        assert alg._Egs == approx(-108.48042111794851, abs=1.0e-12)
        assert alg._E_mmcc_mp[0] == approx(-108.49099104585895, abs=1.0e-12)
        assert alg._E_mmcc_en[0] == approx(-108.49255913217682, abs=1.0e-12)
