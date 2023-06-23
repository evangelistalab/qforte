from pytest import approx
from qforte import system_factory
import numpy as np

def test_json_dump():
    geom = [
            ('H', (0.0, 0.0, 0.0)),
            ('H', (0.0, 0.0, 1.5)),
            ('H', (0.0, 0.0, 3.0)),
            ('H', (0.0, 0.0, 4.5)),
            ('H', (0.0, 0.0, 6.0))
    ]

    N_qubits = 6
    hdim = int(2**N_qubits)

    mol1 = system_factory(build_type = "psi4",
                          mol_geometry = geom, 
                          basis = "sto-3g",
                          multiplicity = 2,
                          json_dump = "h.json",
                          num_frozen_docc = 1,
                          num_frozen_uocc = 1)
    
    H1 = mol1.hamiltonian.sparse_matrix(N_qubits).to_map()
    h1 = np.zeros((hdim,hdim), dtype = 'complex64')
    for i in H1.keys():
        for j in H1[i].keys():
            h1[i,j] = H1[i][j]

    mol2 = system_factory(build_type = "external",
                          filename = "h.json")

    H2 = mol2.hamiltonian.sparse_matrix(N_qubits).to_map()
    h2 = np.zeros((hdim,hdim), dtype = 'complex64')
    for i in H2.keys():
        for j in H2[i].keys():
            h2[i,j] = H2[i][j]

    assert np.linalg.norm(h2 - h1) == approx(0.0)
