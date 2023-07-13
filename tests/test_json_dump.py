from pytest import approx
from qforte import system_factory, sq_op_to_scipy
import scipy
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(THIS_DIR, "h13_dump.json")
class TestJsonDump:
    def test_json_dump(self):
        geom = [("H", (0, 0, r)) for r in range(13)]
        N_qubits = 16        

        mol1 = system_factory(build_type = "psi4",
                              mol_geometry = geom, 
                              basis = "sto-3g",
                              multiplicity = 2,
                              json_dump = data_path,
                              num_frozen_docc = 2,
                              num_frozen_uocc = 3)
    
        H1 = sq_op_to_scipy(mol1.sq_hamiltonian, N_qubits)

        mol2 = system_factory(build_type = "external",
                              filename = data_path)
        
        H2 = sq_op_to_scipy(mol2.sq_hamiltonian, N_qubits)
        
        assert scipy.sparse.linalg.norm(H2 - H1) == approx(0.0, abs = 1.0e-12)

        mol1 = system_factory(build_type = "psi4",
                              mol_geometry = geom, 
                              basis = "sto-3g",
                              multiplicity = 2,
                              json_dump = data_path,
                              num_frozen_docc = 2,
                              num_frozen_uocc = 3,
                              symmetry = "D2h")
    
        H1 = sq_op_to_scipy(mol1.sq_hamiltonian, N_qubits)

        mol2 = system_factory(build_type = "external",
                              filename = data_path)
        
        H2 = sq_op_to_scipy(mol2.sq_hamiltonian, N_qubits)
        
        assert scipy.sparse.linalg.norm(H2 - H1) == approx(0.0, abs = 1.0e-12)