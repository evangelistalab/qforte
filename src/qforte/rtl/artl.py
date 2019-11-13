import qforte
from qforte import vqe
from qforte.rtl import rtl_helpers
from qforte.rtl import artl_helpers
import numpy as np
from scipy import linalg


    ####################################### READ ###########################################
    # This module requires you to pull the mrsqk_pilot branch of QForte, it contains       #
    # the 'perfect_measurement' function as well as the ability to exponeniate operators   #
    # as controlled-unitaries                                                              #
    ########################################################################################

def adaptive_rtl_energy(mol, Nrefs, mr_dt, initial_ref,
                        trot_order = 1,
                        a_el = None,
                        a_sorb = None,
                        use_phase_based_selection=False,
                        use_spin_adapted_refs=False,
                        target_root=None, Ninitial_states=4, inital_dt=1.0, fast=False, var_dt=False,
                        nstates_per_ref=2, print_mats=True, return_all_eigs=False,
                        return_S=False, return_Hbar=False):

    print('\n-----------------------------------------------------')
    print('        Multreference Selected Quantum Krylov   ')
    print('-----------------------------------------------------')

    nqubits = len(initial_ref)
    init_basis_idx = rtl_helpers.ref_to_basis_idx(initial_ref)
    init_basis = qforte.QuantumBasis(init_basis_idx)

    print('\n\n                   ==> MRSQK options <==')
    print('-----------------------------------------------------------')
    print('Initial reference:                       ',  init_basis.str(nqubits))
    print('Dimension of reference space (d):        ',  Nrefs)
    print('Time evolutions per reference (s):       ',  nstates_per_ref-1)
    print('Dimension of Krylov space (N):           ',  Nrefs*nstates_per_ref)
    print('Delta t (in a.u.):                       ',  mr_dt)
    print('Trotter number (m):                      ',  trot_order)
    print('Target root:                             ',  str(target_root))
    print('Use det. selection with sign:            ',  str(use_phase_based_selection))
    print('Use spin adapted references:             ',  str(use_spin_adapted_refs))
    print('Use fast version of algorithm:           ',  str(fast))

    print('\n\n     ==> Initial QK options (for ref. selection)  <==')
    print('-----------------------------------------------------------')
    print('Number of initial time evolutions (s_o): ',  Ninitial_states-1)
    print('Dimension of inital Krylov space (N_o):  ',  Ninitial_states)
    print('Initial delta t_o (in a.u.):             ',  inital_dt)

    if(use_spin_adapted_refs):
        sa_ref_lst = artl_helpers.get_sa_init_ref_lst(initial_ref, Nrefs, Ninitial_states, inital_dt,
                                           mol, target_root=target_root, fast=True,
                                           use_phase_based_selection=use_phase_based_selection)

        nqubits = len(sa_ref_lst[0][0][1])

    else:
        ref_lst = artl_helpers.get_init_ref_lst(initial_ref, Nrefs, Ninitial_states, inital_dt,
                                            mol, target_root=target_root, fast=True,
                                            use_phase_based_selection=use_phase_based_selection)

        nqubits = len(ref_lst[0])

    #NOTE: need get nqubits from Molecule class attribute instead of ref list length
    # Also true for UCC functions
    num_refs = Nrefs
    num_tot_basis = num_refs * nstates_per_ref

    h_mat = np.zeros((num_tot_basis,num_tot_basis), dtype=complex)
    s_mat = np.zeros((num_tot_basis,num_tot_basis), dtype=complex)

    dt_lst = []
    for i in range(Nrefs):
        dt_lst.append(mr_dt)

    if(fast):
        if(use_spin_adapted_refs):
            s_mat, h_mat = rtl_helpers.get_sa_mr_mats_fast(sa_ref_lst, nstates_per_ref,
                                                        dt_lst, mol.get_hamiltonian(),
                                                        nqubits, trot_order=trot_order)

        else:
            s_mat, h_mat = rtl_helpers.get_mr_mats_fast(ref_lst, nstates_per_ref,
                                                        dt_lst, mol.get_hamiltonian(),
                                                        nqubits, trot_order=trot_order)

    else:
        if(use_phase_based_selection or use_spin_adapted_refs):
            raise NotImplementedError("Can't use spin adapted refs or adaptive selection with slow alogrithm.")
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
                                                                        nqubits, mol.get_hamiltonian(),
                                                                        trot_order=trot_order)
                            h_mat[q][p] = np.conj(h_mat[p][q])

                            s_mat[p][q] = rtl_helpers.mr_matrix_element(ref_I, ref_J, dt_I, dt_J,
                                                                        m, n, mol.get_hamiltonian(),
                                                                        nqubits,
                                                                        trot_order=trot_order)
                            s_mat[q][p] = np.conj(s_mat[p][q])



    if(print_mats):
        print('\n\n                ==> MRSQK matricies <==')
        print('-----------------------------------------------------------')

        print("\nS:\n")
        rtl_helpers.matprint(s_mat)
        print('\nk(S): ', np.linalg.cond(s_mat))

        print("\nHbar:\n")
        rtl_helpers.matprint(h_mat)

    evals, evecs = rtl_helpers.canonical_geig_solve(s_mat, h_mat)
    evals_sorted = np.sort(evals)

    if(np.abs(np.imag(evals_sorted[0])) < 1.0e-3):
        Eo = np.real(evals_sorted[0])
    elif(np.abs(np.imag(evals_sorted[1])) < 1.0e-3):
        print('Warning: problem may be ill condidtiond, evals have imaginary components')
        Eo = np.real(evals_sorted[1])
    else:
        print('Warding: problem may be extremely ill conditioned, check evals and k(S)')
        Eo = 0.0

    print('\n\n                   ==> MRSQK summary <==')
    print('-----------------------------------------------------------')
    cs_str = '{:.2e}'.format(np.linalg.cond(s_mat))
    print('Condition number of overlap mat k(S):     ', cs_str)
    print('Final MRSQK Energy:                      ', round(Eo, 10))

    if(return_all_eigs or return_S or return_Hbar):
        return_list = [Eo]
        if(return_all_eigs):
            return_list.append(evals_sorted)
        if(return_S):
            return_list.append(s_mat)
        if(return_Hbar):
            return_list.append(h_mat)

        return return_list

    return Eo
