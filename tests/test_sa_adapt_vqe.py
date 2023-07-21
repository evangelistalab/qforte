from pytest import approx
from qforte import ADAPTVQE
from qforte import system_factory
from qforte import build_effective_operator
from qforte import sq_op_to_scipy
from qforte import ritz_eigh
from qforte import total_spin_z
from qforte import total_number
from qforte import cisd_manifold
import os
import numpy as np
import pytest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(THIS_DIR, 'H4-sto6g-075a.json')

class TestSAADAPTVQE:
    def test_LiH_SA_adapt_vqe(self):
        print('\n')
        geom = [("Li", (0, 0, 0)), ("H", (0, 0, 1))]
        mol = system_factory(system_type = 'molecule',
                             mol_geometry = geom,
                             build_type = 'psi4',
                             basis = 'sto-6g',
                             dipole = True,
                             num_frozen_docc = 1,
                             num_frozen_uocc = 1)
        
        refs = [mol.hf_reference] + cisd_manifold(mol.hf_reference)
        np.random.seed(0)
        weights = np.random.random((len(refs)))
        weights /= np.sum(weights)
        weights = list(weights)

        alg = ADAPTVQE(mol,
                       print_summary_file = False,
                       is_multi_state = True,
                       reference = refs,
                       weights = weights)
        
        alg.run(avqe_thresh = 1e-6,
                pool_type = 'GSD',
                opt_thresh = 1e-7,
                opt_maxiter = 1000,
                adapt_maxiter = 1)

        H = sq_op_to_scipy(mol.sq_hamiltonian, alg._nqb).todense()
        Sz = sq_op_to_scipy(total_spin_z(alg._nqb, do_jw = False), alg._nqb).todense()
        N = sq_op_to_scipy(total_number(alg._nqb, do_jw = False), alg._nqb).todense() 
        dip_x_arr = sq_op_to_scipy(mol.sq_dipole_x, alg._nqb).todense()
        dip_y_arr = sq_op_to_scipy(mol.sq_dipole_y, alg._nqb).todense()
        dip_z_arr = sq_op_to_scipy(mol.sq_dipole_z, alg._nqb).todense()

        H_penalized = H + 1000*Sz@Sz + 1000*(N@N - 4*N + 4*np.eye(H.shape[0]))
        w, v = np.linalg.eigh(H_penalized)

        U = alg.build_Uvqc(amplitudes = alg._tamps)
        Es, A, dip_x, dip_y, dip_z = ritz_eigh(mol.hamiltonian, U, ops_to_compute = [mol.dipole_x, mol.dipole_y, mol.dipole_z])
        total_dip = np.sqrt(np.square(dip_x) + np.square(dip_y) + np.square(dip_z))
        
        for i in range(len(Es)):             
             assert Es[i] == approx(w[i], abs = 1.0e-10)
        
        non_degens = [0,1,2,9,10,15]
        
        dip_dir = np.zeros((len(Es), len(Es)),dtype = "complex")
        for i in non_degens:
             for op in [dip_x_arr, dip_y_arr, dip_z_arr]:
                  sig = op@v[:,i]
                  for j in non_degens:
                       dip_dir[i,j] += (sig.T.conj()@v[:,j])[0,0]**2
        dip_dir = np.sqrt(dip_dir.real)
        
        for i in non_degens:
             for j in non_degens: 
                  assert dip_dir[i,j]-total_dip[i,j] == approx(0.0, abs = 1e-10)
                  
        
        
        

