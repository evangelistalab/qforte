from pytest import approx
from qforte import ADAPTVQE
from qforte import system_factory
from qforte import cisd_manifold
from qforte import build_refprep
from qforte import sq_op_to_scipy
from qforte import q_sc_eom
import numpy as np

import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(THIS_DIR, "H4-sto6g-075a.json")


class TestQSCEOM:
    def test_lih_qsceom(self):
        geom = [("Li", (0, 0, 0)), ("H", (0, 0, 1))]
        mol = system_factory(
            system_type="molecule",
            mol_geometry=geom,
            build_type="psi4",
            basis="sto-6g",
            num_frozen_docc=1,
            num_frozen_uocc=1,
            dipole=True,
            symmetry="C1",
        )
        alg = ADAPTVQE(mol, print_summary_file=False, compact_excitations=True)

        alg.run(
            adapt_maxiter=1000,
            avqe_thresh=0,
            opt_thresh=1e-16,
            pool_type="GSD",
            opt_maxiter=100000,
        )

        U_ansatz = alg.ansatz_circuit(alg._tamps)
        U_hf = build_refprep(mol.hf_reference)
        U_hf.add_circuit(U_ansatz)

        cisd = [
            build_refprep(det)
            for det in cisd_manifold(mol.hf_reference, mol.orb_irreps_to_int)
        ]
        manifold = []
        for i in range(len(cisd)):
            det = cisd[i]
            det.add_circuit(U_ansatz)
            manifold.append(det)

        all_Es, A_plus_ref, ops = q_sc_eom(
            alg._nqb,
            mol.hamiltonian,
            U_hf,
            manifold,
            [mol.dipole_x, mol.dipole_y, mol.dipole_z],
        )
        dip_x, dip_y, dip_z = ops

        total_dip = np.zeros(dip_x.shape)
        for op in [dip_x, dip_y, dip_z]:
            total_dip += np.multiply(op.conj(), op).real
        total_dip = np.sqrt(total_dip)

        H = sq_op_to_scipy(mol.sq_hamiltonian, alg._nqb, N=2, Sz=0).todense()
        fci_dip_x = sq_op_to_scipy(mol.sq_dipole_x, alg._nqb).todense()
        fci_dip_y = sq_op_to_scipy(mol.sq_dipole_y, alg._nqb).todense()
        fci_dip_z = sq_op_to_scipy(mol.sq_dipole_z, alg._nqb).todense()

        w, v = np.linalg.eigh(H)

        for i in range(len(all_Es)):
            assert all_Es[i] - w[i] == approx(0.0, abs=1.0e-10)

        non_degens = [0, 1, 2, 9, 10, 15]

        total_dip = np.zeros(dip_x.shape)
        for op in [dip_x, dip_y, dip_z]:
            total_dip += np.multiply(op.conj(), op).real
        total_dip = np.sqrt(total_dip)

        dip_dir = np.zeros((len(all_Es), len(all_Es)))
        for i in non_degens:
            for op in [fci_dip_x, fci_dip_y, fci_dip_z]:
                sig = op @ v[:, i]
                for j in non_degens:
                    dip_dir[i, j] += (
                        (sig.T.conj() @ v[:, j])[0, 0] * (v[:, j].T.conj() @ sig)[0, 0]
                    ).real
        dip_dir = np.sqrt(dip_dir.real)

        for i in non_degens:
            for j in non_degens:
                assert dip_dir[i, j] - total_dip[i, j] == approx(0.0, abs=1.0e-6)
