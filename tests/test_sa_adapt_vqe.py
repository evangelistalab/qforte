from pytest import approx
from qforte import ADAPTVQE
from qforte import system_factory
from qforte import build_effective_operator
from qforte import sq_op_to_scipy
import os
import numpy as np
import pytest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(THIS_DIR, 'H4-sto6g-075a.json')

class TestSAADAPTVQE:

    @pytest.mark.long
    def test_H4_SA_adapt_vqe_two_states(self):
        print('\n')

        mol = system_factory(system_type = 'molecule',
                             build_type = 'external',
                             basis = 'sto-6g',
                             filename = data_path)

        N_qb = 8

        H = sq_op_to_scipy(mol.sq_hamiltonian, N_qb)
        Es_exact, C = np.linalg.eigh(H.todense())
        
        
        alg = ADAPTVQE(mol,
                       print_summary_file = False,
                       is_multi_state = True,
                       reference = [[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 1, 1, 0, 0, 0]],
                       weights = [0.7, 0.3])
        
        alg.run(avqe_thresh = 1e-6,
                pool_type = 'GSD',
                opt_thresh = 1e-7,
                opt_maxiter = 1000,
                adapt_maxiter = 1000)
         
        H_eff = build_effective_operator(mol.hamiltonian, alg.build_Uvqc())
        Es, C = np.linalg.eigh(H_eff)
        
        assert Es[0] == approx(Es_exact[0], abs=5.0e-11)
        assert Es[1] == approx(Es_exact[1], abs=5.0e-11)
        
        

