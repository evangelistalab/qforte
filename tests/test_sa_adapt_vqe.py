from pytest import approx
from qforte import ADAPTVQE
from qforte import system_factory

from qforte import sq_op_to_scipy
from qforte import ritz_eigh
from qforte import total_spin_z
from qforte import total_number
from qforte import cisd_manifold

import numpy as np

class TestSAADAPTVQE:
     def test_LiH_SA_adapt_vqe(self):
          print('\n')
          geom = [("Li", (0, 0, 0)), ("H", (0, 0, 1))]
          mol = system_factory(system_type = 'molecule',
                               mol_geometry = geom,
                               build_type = 'psi4',
                               basis = 'sto-3g',
                               dipole = True,
                               num_frozen_docc = 1,
                               num_frozen_uocc = 1,
                               symmetry = "C2v",
                               compact_excitations = True)
        
          refs = [mol.hf_reference] + cisd_manifold(mol.hf_reference, mol.orb_irreps_to_int)
          weights = [2**(-i) for i in range(1, len(refs)+1)]
          weights[-1] += 2**(-len(weights))

          alg = ADAPTVQE(mol,
                         print_summary_file = False,
                         is_multi_state = True,
                         reference = refs,
                         weights = weights,
                         compact_excitations = True)
        
          alg.run(avqe_thresh = 1e-12,
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

          H_penalized = H + 100*Sz@Sz + 100*(N@N - 4*N + 4*np.eye(H.shape[0]))
          w, v = np.linalg.eigh(H_penalized)
          w_idx = [0,2,10,15]
          w = w[w_idx]
          v = v[:,w_idx]
          
          U = alg.build_Uvqc(amplitudes = alg._tamps)
           
          Es, A, dip_x, dip_y, dip_z = ritz_eigh(alg._nqb, mol.hamiltonian, U, ops_to_compute = [mol.dipole_x, mol.dipole_y, mol.dipole_z])
     
          E_idx = [0,2,3,5]
          Es = Es[E_idx]
          
          total_dip = np.zeros(dip_x.shape)
          for op in [dip_x, dip_y, dip_z]:
               total_dip += np.multiply(op.conj(),op).real
          total_dip = np.sqrt(total_dip)
          
          total_dip = total_dip[np.ix_(E_idx,E_idx)] 
          
          for i in range(len(Es)):             
               assert Es[i] == approx(w[i], abs = 1.0e-10)
        
          dip_dir = np.zeros((len(Es), len(Es)))
          for i in range(len(Es)):
               for op in [dip_x_arr, dip_y_arr, dip_z_arr]:
                    sig = op@v[:,i]
                    for j in range(len(Es)):
                         dip_dir[i,j] += ((sig.T.conj()@v[:,j])[0,0] * (v[:,j].T.conj()@sig)[0,0]).real
                         
          dip_dir = np.sqrt(dip_dir)      
          
          for i in range(len(Es)):
               for j in range(len(Es)):
                    assert dip_dir[i,j]-total_dip[i,j] == approx(0.0, abs = 1e-10)
                    



        
        

