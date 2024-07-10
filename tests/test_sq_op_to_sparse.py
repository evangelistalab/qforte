from pytest import approx
from qforte import system_factory
from qforte import sq_op_to_scipy
from qforte import total_spin_z
from qforte import total_number

import numpy as np


class TestOpToSparse:
    def test_sq_op_to_scipy(self):
        geom = [("Li", (0, 0, 0)), ("Be", (0, 0, 1))]
        N_qubits = 8
        hdim = 1 << N_qubits

        mol = system_factory(
            build_type="psi4",
            mol_geometry=geom,
            basis="sto-6g",
            multiplicity=2,
            num_frozen_docc=2,
            num_frozen_uocc=4,
            charge=0,
        )

        H = sq_op_to_scipy(mol.sq_hamiltonian, N_qubits).todense()

        hmap = mol.hamiltonian.sparse_matrix(N_qubits).to_map()
        H_slow = np.zeros((hdim, hdim), dtype="complex")
        for i in hmap.keys():
            for j in hmap[i].keys():
                H_slow[i, j] = hmap[i][j]

        assert np.linalg.norm(H_slow - H) == approx(0.0, abs=1.0e-7)

        Sz = sq_op_to_scipy(total_spin_z(N_qubits, False), N_qubits).todense()
        N = sq_op_to_scipy(total_number(N_qubits, False), N_qubits).todense()
        for i in [-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5]:
            for j in range(0, 8):
                H_pen = (
                    H
                    + 1000 * np.square((Sz - i * np.eye(hdim)))
                    + 1000 * np.square((N - j * np.eye(hdim)))
                )
                w, v = np.linalg.eigh(H_pen)
                H_sym = sq_op_to_scipy(
                    mol.sq_hamiltonian, N_qubits, N=j, Sz=i
                ).todense()
                wsym, vsym = np.linalg.eigh(H_sym)
                del vsym
                w_sym_nonzero = [k for k in wsym if abs(k) > 1e-12]
                for k in range(len(w_sym_nonzero)):
                    assert w_sym_nonzero[k] == approx(w[k], abs=1e-7)
