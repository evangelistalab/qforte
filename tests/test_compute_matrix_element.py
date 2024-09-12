from pytest import approx
from qforte import system_factory, Circuit, gate, compute_operator_matrix_element
import numpy as np


class TestComputeMatrixElement:
    def test_H2_Hmat_diagonalization(self):
        to_angs = 0.529177210903

        mol = system_factory(
            system_type="molecule",
            build_type="psi4",
            basis="sto-6g",
            mol_geometry=[("H", (0, 0, 0)), ("H", (0, 0, 1.401 * to_angs))],
            symmetry="d2h",
            multiplicity=1,
            charge=0,
            num_frozen_docc=0,
            num_frozen_uocc=0,
            run_mp2=1,
            run_ccsd=0,
            run_cisd=0,
            run_fci=1,
        )

        circ_hf = Circuit()
        for i in range(2):
            circ_hf.add(gate("X", i))

        circ_doubles = Circuit()
        for i in range(2, 4):
            circ_doubles.add(gate("X", i))

        S_mat = np.zeros((2, 2))
        H_mat = np.zeros((2, 2))

        for i, det_i in enumerate([circ_hf, circ_doubles]):
            for j, det_j in enumerate([circ_hf, circ_doubles]):
                S_mat[i, j] = np.real(
                    compute_operator_matrix_element(
                        mol.hamiltonian.num_qubits(), det_i, det_j
                    )
                )
                H_mat[i, j] = np.real(
                    compute_operator_matrix_element(
                        mol.hamiltonian.num_qubits(), det_i, det_j, mol.hamiltonian
                    )
                )

        assert S_mat[0, 0] == 1
        assert S_mat[0, 1] == 0
        assert S_mat[1, 0] == 0
        assert S_mat[1, 1] == 1

        eig_val, _ = np.linalg.eigh(H_mat)

        assert eig_val[0] == approx(mol.fci_energy, abs=1.0e-12)
        assert eig_val[1] == approx(0.47298335127441565, abs=1.0e-12)
