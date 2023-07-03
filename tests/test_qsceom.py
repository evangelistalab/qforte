from pytest import approx
from qforte import ADAPTVQE
from qforte import system_factory
from qforte import cisd_manifold
from qforte import build_Uprep
from qforte import sq_op_to_scipy
from qforte import total_spin_z
from qforte import total_number
from qforte import q_sc_eom
import numpy as np

import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(THIS_DIR, 'H4-sto6g-075a.json')

class TestQSCEOM:
    def test_h2_qsceom(self):
        print('\n')

        geom = [("H", (0, 0, 0)), ("H", (0, 0, 1.5))]
        mol = system_factory(system_type = 'molecule',
                             mol_geometry = geom,
                             build_type = 'psi4',
                             basis = 'sto-6g')
        
        alg = ADAPTVQE(mol, print_summary_file = False)

        alg.run(adapt_maxiter = 100,
                avqe_thresh = 1e-6,
                opt_thresh = 1e-10,
                pool_type = 'GSD')
        
        U_ansatz = alg.ansatz_circuit(alg._tamps)
        U_hf = build_Uprep(mol.hf_reference, 'occupation_list')
        #Unitaries to build CISD determinants
        U_man = [build_Uprep(det, 'occupation_list') for det in cisd_manifold(mol.hf_reference)]

        #Actual q-sc-EOM Calculation
        E0, Eks = q_sc_eom(mol.hamiltonian, U_ansatz, U_hf, U_man)
        all_Es = [E0] + list(Eks)

        #Get FCI solutions:
        H = sq_op_to_scipy(mol.sq_hamiltonian, alg._nqb).todense()
        Sz = sq_op_to_scipy(total_spin_z(alg._nqb, do_jw = False), alg._nqb).todense()
        N = sq_op_to_scipy(total_number(alg._nqb, do_jw = False), alg._nqb).todense()
        H_penalized = H + 1000*Sz@Sz + (N@N - 4*N + 4*np.eye(H.shape[0]))
        w, v = np.linalg.eigh(H_penalized)
        print(w)
        print(all_Es)
        #Check ground and excited states:
        for i in range(len(all_Es)):
            assert all_Es[i] - w[i] == approx(0.0, abs = 1.0e-12)
         

        