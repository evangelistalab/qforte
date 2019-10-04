import qforte
from qforte.rtl import rtl_helpers
from qforte.rtl import artl_helpers
# from qforte.utils import transforms
# from qforte.utils import trotterization as trot

import numpy as np
from scipy import linalg


    ####################################### READ ###########################################
    # This module requires you to pull the rtl_pilot branch of QForte, it contains         #
    # the 'perfect_measurement' function as well as the ability to exponeniate operators   #
    # as controlled-unitaries                                                              #
    ########################################################################################

def adaptive_rtl_energy(mol, Nrefs, mr_dt, initial_ref, target_root=None, Ninitial_states=4, inital_dt=1.0, fast=False, var_dt=False,
                        nstates_per_ref=2, print_mats=True, return_all_eigs=False,
                        return_S=False, return_Hbar=False):

    # Below instructions will be executed by a function that returns a vector of reffs.
    ref_lst = artl_helpers.get_init_ref_lst(initial_ref, Nrefs, Ninitial_states, inital_dt,
                                            mol, target_root=target_root, fast=True)

    #NOTE: need get nqubits from Molecule class attribute instead of ref list length
    # Also true for UCC functions
    num_refs = Nrefs
    num_tot_basis = num_refs * nstates_per_ref
    nqubits = len(ref_lst[0])

    h_mat = np.zeros((num_tot_basis,num_tot_basis), dtype=complex)
    s_mat = np.zeros((num_tot_basis,num_tot_basis), dtype=complex)

    dt_lst = []
    for i in range(Nrefs):
        dt_lst.append(mr_dt)

    if(fast):
        print('using faster fast algorithm lol')
        s_mat, h_mat = rtl_helpers.get_mr_mats_fast(ref_lst, nstates_per_ref,
                                                    dt_lst, mol.get_hamiltonian(),
                                                    nqubits)

    else:
        for I in range(num_refs):
            for J in range(num_refs):
                for m in range(nstates_per_ref):
                    for n in range(nstates_per_ref):
                        p = I*nstates_per_ref + m
                        q = J*nstates_per_ref + n
                        if(q>=p):
                            ref_I = ref_lst[I]
                            ref_J = ref_lst[J]
                            dt_I = dt_lst[I]
                            dt_J = dt_lst[J]

                            h_mat[p][q] = rtl_helpers.mr_matrix_element(ref_I, ref_J, dt_I, dt_J,
                                                                        m, n, mol.get_hamiltonian(),
                                                                        nqubits, mol.get_hamiltonian())
                            h_mat[q][p] = np.conj(h_mat[p][q])

                            s_mat[p][q] = rtl_helpers.mr_matrix_element(ref_I, ref_J, dt_I, dt_J,
                                                                        m, n, mol.get_hamiltonian(),
                                                                        nqubits)
                            s_mat[q][p] = np.conj(s_mat[p][q])



    if(print_mats):
        print('------------------------------------------------')
        print('   Matricies for MR Quantum Real-Time Lanczos')
        print('------------------------------------------------')
        print('Nrefs:             ', num_refs)
        print('Nevos per ref:     ', nstates_per_ref)
        print('Ntot states  :     ', num_tot_basis)
        print('Delta t list :     ', dt_lst)

        print("\nS:\n")
        rtl_helpers.matprint(s_mat)

        print('\nk(S): ', np.linalg.cond(s_mat))

        print("\nHbar:\n")
        rtl_helpers.matprint(h_mat)

        print('\nk(Hbar): ', np.linalg.cond(h_mat))


    # evals, evecs = linalg.eig(h_mat,s_mat)
    evals, evecs = rtl_helpers.canonical_geig_solve(s_mat, h_mat)
    print('\nRTLanczos (unsorted!) evals from measuring ancilla:\n', evals)
    # print('type of evals list: ', type(evals))

    evals_sorted = np.sort(evals)

    if(np.abs(np.imag(evals_sorted[0])) < 1.0e-3):
        Eo = np.real(evals_sorted[0])
    elif(np.abs(np.imag(evals_sorted[1])) < 1.0e-3):
        print('Warning: problem may be ill condidtiond, evals have imaginary components')
        Eo = np.real(evals_sorted[1])
    else:
        print('Warding: problem may be extremely ill conditioned, check evals and k(S)')
        Eo = 0.0

    if(return_all_eigs or return_S or return_Hbar):
        return_list = [Eo]
        if(return_all_eigs):
            return_list.append(evals)
        if(return_S):
            return_list.append(s_mat)
        if(return_Hbar):
            return_list.append(h_mat)

        return return_list

    return Eo
