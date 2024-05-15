from pytest import approx
from qforte import QubitOperator, smart_print, system_factory, SPQE, UCCNPQE

import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(THIS_DIR, "H4-sto6g-075a.json")


class TestSPQE:
    def test_H4_spqe_exact(self):
        print("\n")

        # The FCI energy for H4 at 0.75 Angstrom in a sto-6g basis
        Efci = -2.1628978832666865
        # The Nuclear repulsion energy
        Enuc = 3.057468328315556

        mol = system_factory(
            stytem_type="molecule",
            build_type="external",
            basis="sto-6g",
            filename=data_path,
        )

        Hnonzero = QubitOperator()
        for term in mol._hamiltonian.terms():
            if abs(term[0]) > 1.0e-14:
                Hnonzero.add(term[0], term[1])
        mol._hamiltonian = Hnonzero
        smart_print(Hnonzero)

        alg = SPQE(mol, print_summary_file=False)
        alg.run(spqe_maxiter=20, spqe_thresh=1.0e-4, opt_thresh=1.0e-5, dt=0.0001)

        Egs_elec = alg.get_gs_energy()
        # Egs = Egs_elec + Enuc
        Egs = Egs_elec
        assert Egs == approx(Efci, abs=5.0e-11)

    def test_spqe_max_excit_rank(self):
        Rhh = 1.5

        mol = system_factory(
            system_type="molecule",
            build_type="psi4",
            basis="sto-6g",
            mol_geometry=[
                ("H", (0, 0, -3 * Rhh / 2)),
                ("H", (0, 0, -Rhh / 2)),
                ("H", (0, 0, Rhh / 2)),
                ("H", (0, 0, 3 * Rhh / 2)),
            ],
            symmetry="d2h",
            multiplicity=1,  # Only singlets will work with QForte
            charge=0,
            num_frozen_docc=0,
            num_frozen_uocc=0,
            run_mp2=0,
            run_ccsd=0,
            run_cisd=0,
            run_fci=0,
        )

        uccnpqe_sd = UCCNPQE(mol, verbose=False)
        uccnpqe_sd.run(pool_type="SD", opt_thresh=1.0e-5, opt_maxiter=50)

        spqe_sd = SPQE(mol, verbose=False)
        spqe_sd.run(
            spqe_thresh=0,
            spqe_maxiter=1,
            opt_maxiter=50,
            opt_thresh=1.0e-5,
            use_cumulative_thresh=True,
            max_excit_rank=2,
        )

        assert len(spqe_sd._pool_obj) == len(uccnpqe_sd._pool_obj)
        # Although both algorithms utilize ansatze containing the same operators,
        # their ordering is different. Thus the UCCSD energies resulting from the
        # SPQE and UCCNPQE algorithms will not be identical
        assert spqe_sd._Egs == approx(uccnpqe_sd._Egs, abs=1.0e-5)

    def test_spqe_scipy_solver(self):
        # In this test, we confirm that the SPQE algorithm produces
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

        jacobi = SPQE(
            mol, compact_excitations=True, qubit_excitations=False, diis_max_dim=8
        )
        jacobi.run(optimizer="jacobi", opt_maxiter=50)

        bfgs = SPQE(mol, compact_excitations=True, qubit_excitations=False)
        bfgs.run(optimizer="BFGS", opt_maxiter=50)

        assert jacobi.get_gs_energy() == approx(bfgs.get_gs_energy(), abs=1.0e-8)
