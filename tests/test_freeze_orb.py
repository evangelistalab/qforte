import pytest
from pytest import approx
from qforte import system_factory, UCCNVQE, ADAPTVQE, UCCNPQE, SPQE


class TestFreezingOrbitals:
    @pytest.mark.parametrize(
        "method, options",
        [
            (UCCNVQE, {"pool_type": "SDTQ"}),
            (ADAPTVQE, {"pool_type": "SDTQ", "avqe_thresh": 1.0e-3}),
            (UCCNPQE, {"pool_type": "SDTQ"}),
            (SPQE, {"spqe_thresh": 1.0e-4, "dt": 0.0001}),
        ],
    )
    def test_freeze_orb_ucc(self, method, options):
        mol = system_factory(
            system_type="molecule",
            build_type="psi4",
            basis="sto-3g",
            mol_geometry=[("Be", (0, 0, -1.2)), ("Be", (0, 0, 1.2))],
            symmetry="d2h",
            num_frozen_docc=2,
            num_frozen_uocc=3,
        )

        alg = method(mol)

        alg.run(**options)

        Egs = alg.get_gs_energy()
        # WARNING: Due to a bug in Psi4, the energies stored in the fci_energy attrbitute of the Molecule class are not
        #          correct when the number of frozen virtual orbitals is larger than zero.
        Efci = -28.747184707540754

        assert Egs == approx(Efci, abs=1.0e-10)
