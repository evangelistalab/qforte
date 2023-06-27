from pytest import approx
from qforte import system_factory, sq_op_to_scipy
import numpy as np
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(THIS_DIR, "h5_dump.json")
class TestOpToSparse:
    def test_sq_op_to_scipy(self):
        geom = [("H", (0, 0, r)) for r in range(9)]
        N_qubits = 8
        hdim = int(2**N_qubits) 

        mol1 = system_factory(build_type = "psi4",
                              mol_geometry = geom, 
                              basis = "sto-3g",
                              multiplicity = 2,
                              json_dump = data_path,
                              num_frozen_docc = 2,
                              num_frozen_uocc = 3)
    
        H = sq_op_to_scipy(mol1.sq_hamiltonian, N_qubits).todense()

        hmap = mol1.hamiltonian.sparse_matrix(N_qubits).to_map()
        H_slow = np.zeros((hdim,hdim), dtype = 'complex')
        for i in hmap.keys():
            for j in hmap[i].keys():
                H_slow[i,j] = hmap[i][j]
        
        assert np.linalg.norm(H_slow - H) == approx(0.0, abs = 1.0e-12)