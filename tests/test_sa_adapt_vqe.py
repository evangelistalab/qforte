from pytest import approx
from qforte import ADAPTVQE
from qforte import system_factory
from qforte import build_effective_operator
from qforte import sq_op_to_scipy
from qforte import ritz_eigh
from qforte import total_spin_z
from qforte import total_number


import os
import numpy as np
import pytest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(THIS_DIR, 'H4-sto6g-075a.json')

class TestSAADAPTVQE:

    @pytest.mark.long
    def test_H4_SA_adapt_vqe_two_states(self):

        print('\n')
        geom = [("H", (0, 0, r)) for r in range(4)]
        mol = system_factory(system_type = 'molecule',
                             mol_geometry = geom,
                             build_type = 'psi4',
                             basis = 'sto-6g')
        
        refs = [[1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 1, 1, 0, 0]]
         
        alg = ADAPTVQE(mol,
                       print_summary_file = False,
                       is_multi_state = True,
                       reference = refs,
                       weights = [0.7, 0.3])
        
        alg.run(avqe_thresh = 1e-6,
                pool_type = 'GSD',
                opt_thresh = 1e-7,
                opt_maxiter = 1000,
                adapt_maxiter = 1000)

        H = sq_op_to_scipy(mol.sq_hamiltonian, alg._nqb).todense()
        Sz = sq_op_to_scipy(total_spin_z(alg._nqb, do_jw = False), alg._nqb).todense()
        N = sq_op_to_scipy(total_number(alg._nqb, do_jw = False), alg._nqb).todense()
        H_penalized = H + 1000*Sz@Sz + 1000*(N@N - 8*N + 16*np.eye(H.shape[0]))
        w, v = np.linalg.eigh(H_penalized)


        U = alg.build_Uvqc(amplitudes = alg._tamps)
        Es, A = ritz_eigh(mol.hamiltonian, U)
        print(Es)
        print(w) 
        assert Es[0] == approx(w[0], abs = 1.0e-10)
        assert Es[1] == approx(w[2], abs = 1.0e-10)
        
        

