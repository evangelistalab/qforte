from pytest import approx
from qforte import ADAPTVQE
from qforte import system_factory
from qforte import sq_op_to_scipy
from qforte import ritz_eigh
from qforte import cisd_manifold
from qforte import build_Uprep

import numpy as np

class TestSAADAPTVQE:
     def test_LiH_SA_adapt_vqe(self):
          geom = [("Li", (0, 0, 0)), ("H", (0, 0, 1))]
          mol = system_factory(system_type = 'molecule',
                               mol_geometry = geom,
                               build_type = 'psi4',
                               basis = 'sto-3g',
                               dipole = True,
                               num_frozen_docc = 1,
                               num_frozen_uocc = 1,
                               symmetry = "C2v")
        
          refs = [mol.hf_reference] + cisd_manifold(mol.hf_reference)
          weights = [2**(-i-1) for i in range(len(refs))]
          weights[-1] += 2**(-len(weights))
          
          alg = ADAPTVQE(mol,
                         print_summary_file = False,
                         is_multi_state = True,
                         reference = refs,
                         weights = weights,
                         compact_excitations = True)
        
          H = sq_op_to_scipy(mol.sq_hamiltonian, alg._nqb, Sz = 0, N = 2).todense()
          
          w, v = np.linalg.eigh(H)
          
          non_degens = [0, 1, 2, 7, 8, 15]
          w = w[non_degens]
          v = v[:,non_degens]
          
          dip_x_arr = sq_op_to_scipy(mol.sq_dipole_x, alg._nqb).todense()
          dip_y_arr = sq_op_to_scipy(mol.sq_dipole_y, alg._nqb).todense()
          dip_z_arr = sq_op_to_scipy(mol.sq_dipole_z, alg._nqb).todense()
          
          alg.run(avqe_thresh = 1e-12,
                  pool_type = 'GSD',
                  opt_thresh = 1e-7,
                  opt_maxiter = 1000,
                  adapt_maxiter = 1)
          
          U = alg.build_Uvqc(amplitudes = alg._tamps)
           
          Es, A, ops = ritz_eigh(alg._nqb, mol.hamiltonian, U, [mol.dipole_x, mol.dipole_y, mol.dipole_z])
          dip_x, dip_y, dip_z = ops
          
          Es = Es[non_degens]
          
          
          for i in range(len(Es)):             
               assert Es[i] == approx(w[i], abs = 1.0e-10)
          
          total_dip = np.zeros(dip_x.shape)
          for op in [dip_x, dip_y, dip_z]:
               total_dip += np.multiply(op.conj(),op).real
          total_dip = np.sqrt(total_dip) 
          total_dip = total_dip[np.ix_(non_degens,non_degens)] 
           
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
                    
          circ_refs = [build_Uprep(ref, "occupation_list") for ref in refs]

          alg = ADAPTVQE(mol,
               print_summary_file = False,
               is_multi_state = True,
               reference = circ_refs,
               weights = weights,
               compact_excitations = True,
               state_prep_type = "unitary_circ")
        
          alg.run(avqe_thresh = 1e-12,
                  pool_type = 'GSD',
                  opt_thresh = 1e-7,
                  opt_maxiter = 1000,
                  adapt_maxiter = 1)

          U = alg.build_Uvqc(amplitudes = alg._tamps)
           
          Es, A, ops = ritz_eigh(alg._nqb, mol.hamiltonian, U, [mol.dipole_x, mol.dipole_y, mol.dipole_z]) 
          dip_x, dip_y, dip_z = ops
          Es = Es[non_degens]
          
          for i in range(len(Es)):             
               assert Es[i] == approx(w[i], abs = 1.0e-10)
          
          total_dip = np.zeros(dip_x.shape)
          for op in [dip_x, dip_y, dip_z]:
               total_dip += np.multiply(op.conj(),op).real
          total_dip = np.sqrt(total_dip)
          total_dip = total_dip[np.ix_(non_degens,non_degens)] 
          
          for i in range(len(Es)):
               for j in range(len(Es)):
                    assert dip_dir[i,j]-total_dip[i,j] == approx(0.0, abs = 1e-10)

          alg = ADAPTVQE(mol,
               print_summary_file = False,
               is_multi_state = True,
               reference = refs,
               weights = weights,
               compact_excitations = False)
        
          alg.run(avqe_thresh = 1e-12,
                  pool_type = 'GSD',
                  opt_thresh = 1e-7,
                  opt_maxiter = 1000,
                  adapt_maxiter = 1)

          U = alg.build_Uvqc(amplitudes = alg._tamps)
           
          Es, A, ops = ritz_eigh(alg._nqb, mol.hamiltonian, U, [mol.dipole_x, mol.dipole_y, mol.dipole_z])
          dip_x, dip_y, dip_z = ops
          
          Es = Es[non_degens]
          for i in range(len(Es)):             
               assert Es[i] == approx(w[i], abs = 1.0e-10)
          
          total_dip = np.zeros(dip_x.shape)
          for op in [dip_x, dip_y, dip_z]:
               total_dip += np.multiply(op.conj(),op).real
          total_dip = np.sqrt(total_dip)
          total_dip = total_dip[np.ix_(non_degens,non_degens)] 
          
          for i in range(len(Es)):
               for j in range(len(Es)):
                    assert dip_dir[i,j]-total_dip[i,j] == approx(0.0, abs = 1e-10)
          
     
          
          
          
