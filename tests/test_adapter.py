from pytest import approx
from qforte import system_factory


def test_energies():
    geom = [
        ("H", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, 1.5)),
        ("H", (0.0, 0.0, 3.0)),
        ("H", (0.0, 0.0, 4.5)),
    ]

    mol = system_factory(
        system_type="molecule",
        build_type="psi4",
        basis="sto-3g",
        mol_geometry=geom,
        run_mp2=True,
        run_cisd=True,
        run_ccsd=True,
        run_fci=True,
    )
    assert mol.hf_energy == approx(-1.8291374120321253, abs=1e-6)
    assert mol.mp2_energy == approx(-1.91558883081879, abs=1e-6)
    assert mol.cisd_energy == approx(-1.981824023626, abs=1e-6)
    assert mol.ccsd_energy == approx(-1.997624159882714, abs=1e-6)
    assert mol.fci_energy == approx(-1.996150325300031, abs=1e-6)
