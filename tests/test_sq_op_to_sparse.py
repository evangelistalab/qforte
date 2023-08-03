from pytest import approx
from qforte import system_factory, sq_op_to_scipy
import numpy as np

class TestOpToSparse:
    def test_sq_op_to_scipy(self):
        geom = [("Li", (0, 0, 0)), ("Be", (0, 0, 1))]
        N_qubits = 8
        hdim = 1 << N_qubits

        mol = system_factory(build_type = "psi4",
                              mol_geometry = geom, 
                              basis = "sto-6g",
                              multiplicity = 2,
                              num_frozen_docc = 2,
                              num_frozen_uocc = 4,
                              charge = 0)
    
        H = sq_op_to_scipy(mol.sq_hamiltonian, N_qubits).todense()

        hmap = mol.hamiltonian.sparse_matrix(N_qubits).to_map()
        H_slow = np.zeros((hdim,hdim), dtype = 'complex')
        for i in hmap.keys():
            for j in hmap[i].keys():
                H_slow[i,j] = hmap[i][j]
        
        assert np.linalg.norm(H_slow - H) == approx(0.0, abs = 1.0e-11)