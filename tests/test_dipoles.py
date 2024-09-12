from pytest import approx
from qforte import system_factory, sq_op_to_scipy
import numpy as np
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestDipoles:
    def test_dipoles(self):
        geom = [("H", (0, 0, 0)), ("Li", (0, 0, 1))]
        N_qubits = 8

        mol1 = system_factory(
            build_type="psi4",
            mol_geometry=geom,
            basis="sto-3g",
            multiplicity=1,
            num_frozen_docc=1,
            num_frozen_uocc=1,
            dipole=True,
            symmetry="C1",
        )

        H = sq_op_to_scipy(mol1.sq_hamiltonian, N_qubits).todense()
        mu_x = sq_op_to_scipy(mol1.sq_dipole_x, N_qubits).todense()
        mu_y = sq_op_to_scipy(mol1.sq_dipole_y, N_qubits).todense()
        mu_z = sq_op_to_scipy(mol1.sq_dipole_z, N_qubits).todense()

        E, C = np.linalg.eigh(H)

        inds = [0, 10, 17, 18, 79]
        dipole = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                for op in [mu_x, mu_y, mu_z]:
                    dipole[i, j] += (
                        (C[:, inds[i]].T.conj() @ op @ C[:, inds[j]])[0, 0].real
                    ) ** 2
                dipole[i, j] = np.sqrt(dipole[i, j])

        psi4_dipoles = [1.8751853, 1.6664460, 0.0250378, 0.0250378, 3.2070140]
        psi4_tdms = [0.2422177, 1.2959115, 1.2959115, 0.1294187]
        for i in range(5):
            assert dipole[i, i] - psi4_dipoles[i] == approx(0.0, abs=1e-6)
        for i in range(1, 5):
            assert dipole[0, i] - psi4_tdms[i - 1] == approx(0.0, abs=1e-6)
