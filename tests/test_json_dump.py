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
                              num_frozen_uocc = 3,
                              dipole = True,
                              symmetry = "C1")
    
        H1 = sq_op_to_scipy(mol1.sq_hamiltonian, N_qubits)
        mu_x1 = sq_op_to_scipy(mol1.sq_dipole_x, N_qubits)
        mu_y1 = sq_op_to_scipy(mol1.sq_dipole_y, N_qubits)
        mu_z1 = sq_op_to_scipy(mol1.sq_dipole_z, N_qubits)

        mol2 = system_factory(build_type = "external",
                              filename = data_path)
        
        H2 = sq_op_to_scipy(mol2.sq_hamiltonian, N_qubits)
        mu_x2 = sq_op_to_scipy(mol2.sq_dipole_x, N_qubits)
        mu_y2 = sq_op_to_scipy(mol2.sq_dipole_y, N_qubits)
        mu_z2 = sq_op_to_scipy(mol2.sq_dipole_z, N_qubits)
        
        assert scipy.sparse.linalg.norm(H2 - H1) == approx(0.0, abs = 1.0e-12)
        assert scipy.sparse.linalg.norm(mu_x2 - mu_x1) == approx(0.0, abs = 1.0e-12)
        assert scipy.sparse.linalg.norm(mu_y2 - mu_y1) == approx(0.0, abs = 1.0e-12)
        assert scipy.sparse.linalg.norm(mu_z2 - mu_z1) == approx(0.0, abs = 1.0e-12)

        mol1 = system_factory(build_type = "psi4",
                              mol_geometry = geom, 
                              basis = "sto-3g",
                              multiplicity = 2,
                              json_dump = data_path,
                              num_frozen_docc = 2,
                              num_frozen_uocc = 3,
                              symmetry = "D2h",
                              dipole = True)
    
        H1 = sq_op_to_scipy(mol1.sq_hamiltonian, N_qubits)
        mu_x1 = sq_op_to_scipy(mol1.sq_dipole_x, N_qubits)
        mu_y1 = sq_op_to_scipy(mol1.sq_dipole_y, N_qubits)
        mu_z1 = sq_op_to_scipy(mol1.sq_dipole_z, N_qubits)

        mol2 = system_factory(build_type = "external",
                              filename = data_path)
        
        H2 = sq_op_to_scipy(mol2.sq_hamiltonian, N_qubits)
        mu_x2 = sq_op_to_scipy(mol2.sq_dipole_x, N_qubits)
        mu_y2 = sq_op_to_scipy(mol2.sq_dipole_y, N_qubits)
        mu_z2 = sq_op_to_scipy(mol2.sq_dipole_z, N_qubits)
        
        assert scipy.sparse.linalg.norm(H2 - H1) == approx(0.0, abs = 1.0e-12)
        assert scipy.sparse.linalg.norm(mu_x2 - mu_x1) == approx(0.0, abs = 1.0e-12)
        assert scipy.sparse.linalg.norm(mu_y2 - mu_y1) == approx(0.0, abs = 1.0e-12)
        assert scipy.sparse.linalg.norm(mu_z2 - mu_z1) == approx(0.0, abs = 1.0e-12) 