import numpy as np
from pytest import approx
from qforte import ADAPTVQE, UCCNVQE
from qforte import Circuit, Computer, gate, system_factory

import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(THIS_DIR, "H4-sto6g-075a.json")


class TestComputerInit:
    # @mark.skip(reason="long")
    def test_H4_VQE(self):
        mol = system_factory(
            system_type="molecule",
            build_type="external",
            basis="sto-6g",
            filename=data_path,
        )
        nqubits = len(mol.hf_reference)
        fci_energy = -2.162897881184882

        computer = Computer(nqubits)
        coeff_vec = np.zeros(2**nqubits)
        coeff_vec[int("00001111", 2)] = 1
        coeff_vec[int("00110011", 2)] = 0.2
        coeff_vec[int("00111100", 2)] = 0.1
        coeff_vec[int("11001100", 2)] = 0.04
        coeff_vec /= np.linalg.norm(coeff_vec)
        computer.set_coeff_vec(coeff_vec)

        # Analytic and fin dif gradients agree
        analytic = UCCNVQE(mol, reference=computer, state_prep_type="computer")
        analytic.run(use_analytic_grad=False, pool_type="SD")
        findif = UCCNVQE(mol, reference=computer, state_prep_type="computer")
        findif.run(use_analytic_grad=True, pool_type="SD")
        assert analytic.get_gs_energy() == approx(findif.get_gs_energy(), abs=1.0e-8)

        # Computer-based and non-compute based agree
        hf = ADAPTVQE(mol)
        hf.run(use_analytic_grad=True, pool_type="GSD", avqe_thresh=1e-5)
        comp = ADAPTVQE(mol, reference=computer, state_prep_type="computer")
        comp.run(use_analytic_grad=True, pool_type="GSD", avqe_thresh=1e-5)
        assert hf.get_gs_energy() == approx(comp.get_gs_energy(), abs=1.0e-8)
        assert hf.get_gs_energy() == approx(fci_energy, abs=1.0e-8)
