import qforte
from qforte import vqe
from qforte.rtl import rtl_helpers
from qforte.rtl import artl_helpers
import numpy as np
from scipy import linalg


    ####################################### READ ###########################################
    # This module requires you to pull the rtl_pilot branch of QForte, it contains         #
    # the 'perfect_measurement' function as well as the ability to exponeniate operators   #
    # as controlled-unitaries                                                              #
    ########################################################################################

def adaptive_rtl_energy(mol, Nrefs, mr_dt, initial_ref,
                        trot_order = 1,
                        use_phase_based_selection=False,
                        use_spin_adapted_refs=False,
                        target_root=None, Ninitial_states=4, inital_dt=1.0, fast=False, var_dt=False,
                        nstates_per_ref=2, print_mats=True, return_all_eigs=False,
                        return_S=False, return_Hbar=False):



    if(use_spin_adapted_refs):
        # raise NotImplementedError('Still in template for get_sa_init_ref_lst().')
        sa_ref_lst = artl_helpers.get_sa_init_ref_lst(initial_ref, Nrefs, Ninitial_states, inital_dt,
                                           mol, target_root=target_root, fast=True,
                                           use_phase_based_selection=use_phase_based_selection)

        ##-##
        print('\n\nsa_ref list:')
        print('----------------------------')
        for ref in sa_ref_lst:
            print('  \n', ref)
        print('')
        ##-##
        nqubits = len(sa_ref_lst[0][0][1])
    else:
        ref_lst = artl_helpers.get_init_ref_lst(initial_ref, Nrefs, Ninitial_states, inital_dt,
                                            mol, target_root=target_root, fast=True,
                                            use_phase_based_selection=use_phase_based_selection)

        ##-##
        print('\n\nref list:')
        print('----------------------------')
        for ref in ref_lst:
            print(ref)
        print('')
        ##-##
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
        print('using faster fast algorithm lol')
        if(use_spin_adapted_refs):
            # raise NotImplementedError('Still in template for get_sa_mr_mats_fast().')
            s_mat, h_mat = rtl_helpers.get_sa_mr_mats_fast(sa_ref_lst, nstates_per_ref,
                                                        dt_lst, mol.get_hamiltonian(),
                                                        nqubits, trot_order=trot_order)
            # s_mat, h_mat = rtl_helpers.get_mr_mats_fast(sa_ref_lst, nstates_per_ref,
            #                                             dt_lst, mol.get_hamiltonian(),
            #                                             nqubits)
        else:
            s_mat, h_mat = rtl_helpers.get_mr_mats_fast(ref_lst, nstates_per_ref,
                                                        dt_lst, mol.get_hamiltonian(),
                                                        nqubits, trot_order=trot_order)

    else:
        if(use_phase_based_selection or use_spin_adapted_refs or use_adapt_selection):
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

    evals, evecs = rtl_helpers.canonical_geig_solve(s_mat, h_mat)
    print('\nRTLanczos (unsorted!) evals from measuring ancilla:\n', evals)

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
            return_list.append(evals_sorted)
        if(return_S):
            return_list.append(s_mat)
        if(return_Hbar):
            return_list.append(h_mat)

        return return_list

    return Eo

# def super_adaptive_rtl_energy(mol, Nrefs, mr_dt, initial_ref,
#                         use_phase_based_selection=False,
#                         use_spin_adapted_refs=False,
#                         target_root=None, Ninitial_states=4, inital_dt=1.0, fast=False, var_dt=False,
#                         nstates_per_ref=2, print_mats=True, return_all_eigs=False,
#                         return_S=False, return_Hbar=False):
#
#     if(fast != True):
#         raise NotImplementedError('Slow version of super_adaptive_rtl is not yet implemented.')
#
#     if(use_spin_adapted_refs):
#         raise NotImplementedError('Still in template for get_sa_init_ref_lst().')
#         sa_ref_lst = artl_helpers.get_adaptive_sa_init_ref_lst(initial_ref, Nrefs, Ninitial_states, inital_dt,
#                                            mol, target_root=target_root, fast=True,
#                                            use_phase_based_selection=use_phase_based_selection)
#
#         ##-##
#         print('\n\nsa_ref list:')
#         print('----------------------------')
#         for ref in sa_ref_lst:
#             print(ref)
#         print('')
#         ##-##
#         nqubits = len(sa_ref_lst[0][0][1])
#     else:
#         # ref_lst = artl_helpers.get_adaptive_init_ref_lst(initial_ref, Nrefs, Ninitial_states, inital_dt,
#         #                                     mol, target_root=target_root, fast=True,
#         #                                     use_phase_based_selection=use_phase_based_selection)
#
#         ref_lst = [initial_ref]
#
#         ##-##
#         print('\n\nref list:')
#         print('----------------------------')
#         for ref in ref_lst:
#             print(ref)
#         print('')
#         ##-##
#         nqubits = len(ref_lst[0])
#
#     #NOTE: need get nqubits from Molecule class attribute instead of ref list length
#     # Also true for UCC functions
#     num_refs = Nrefs
#     num_tot_basis = num_refs * nstates_per_ref
#
#
#     h_mat = np.zeros((num_tot_basis,num_tot_basis), dtype=complex)
#     s_mat = np.zeros((num_tot_basis,num_tot_basis), dtype=complex)
#
#     dt_lst = []
#     for i in range(Nrefs):
#         dt_lst.append(mr_dt)
#
#     if(fast):
#         print('using faster fast algorithm lol')
#         if(use_spin_adapted_refs):
#             raise NotImplementedError('Still in template for get_sa_mr_mats_fast().')
#             # s_mat, h_mat = rtl_helpers.get_sa_mr_mats_fast(sa_ref_lst, nstates_per_ref,
#             #                                             dt_lst, mol.get_hamiltonian(),
#             #                                             nqubits)
#             s_mat, h_mat = rtl_helpers.update_sa_mr_mats_fast(sa_ref_lst, nstates_per_ref,
#                                                         dt_lst, mol.get_hamiltonian(),
#                                                         nqubits)
#
#         else:
#             # s_mat, h_mat = rtl_helpers.get_mr_mats_fast(ref_lst, nstates_per_ref,
#             #                                             dt_lst, mol.get_hamiltonian(),
#             #                                             nqubits)
#             s_mat, h_mat, ref_lst = rtl_helpers.update_mr_mats_fast(s_mat, h_mat, ref_lst, nstates_per_ref,
#                                                            dt_lst, mol.get_hamiltonian(),
#                                                            nqubits)
#
#     # else:
#     #     if(use_phase_based_selection or use_spin_adapted_refs or use_adapt_selection):
#     #         raise NotImplementedError("Can't use spin adapted refs or adaptive selection with slow alogrithm.")
#     #     for I in range(num_refs):
#     #         for J in range(num_refs):
#     #             for m in range(nstates_per_ref):
#     #                 for n in range(nstates_per_ref):
#     #                     p = I*nstates_per_ref + m
#     #                     q = J*nstates_per_ref + n
#     #                     if(q>=p):
#     #                         ref_I = ref_lst[I]
#     #                         ref_J = ref_lst[J]
#     #                         dt_I = dt_lst[I]
#     #                         dt_J = dt_lst[J]
#     #
#     #                         h_mat[p][q] = rtl_helpers.mr_matrix_element(ref_I, ref_J, dt_I, dt_J,
#     #                                                                     m, n, mol.get_hamiltonian(),
#     #                                                                     nqubits, mol.get_hamiltonian())
#     #                         h_mat[q][p] = np.conj(h_mat[p][q])
#     #
#     #                         s_mat[p][q] = rtl_helpers.mr_matrix_element(ref_I, ref_J, dt_I, dt_J,
#     #                                                                     m, n, mol.get_hamiltonian(),
#     #                                                                     nqubits)
#     #                         s_mat[q][p] = np.conj(s_mat[p][q])
#
#
#
#     if(print_mats):
#         print('------------------------------------------------')
#         print('   Matricies for MR Quantum Real-Time Lanczos')
#         print('------------------------------------------------')
#         print('Nrefs:             ', num_refs)
#         print('Nevos per ref:     ', nstates_per_ref)
#         print('Ntot states  :     ', num_tot_basis)
#         print('Delta t list :     ', dt_lst)
#
#         print("\nS:\n")
#         rtl_helpers.matprint(s_mat)
#
#         print('\nk(S): ', np.linalg.cond(s_mat))
#
#         print("\nHbar:\n")
#         rtl_helpers.matprint(h_mat)
#
#         print('\nk(Hbar): ', np.linalg.cond(h_mat))
#
#     evals, evecs = rtl_helpers.canonical_geig_solve(s_mat, h_mat)
#     print('\nRTLanczos (unsorted!) evals from measuring ancilla:\n', evals)
#
#     evals_sorted = np.sort(evals)
#
#     if(np.abs(np.imag(evals_sorted[0])) < 1.0e-3):
#         Eo = np.real(evals_sorted[0])
#     elif(np.abs(np.imag(evals_sorted[1])) < 1.0e-3):
#         print('Warning: problem may be ill condidtiond, evals have imaginary components')
#         Eo = np.real(evals_sorted[1])
#     else:
#         print('Warding: problem may be extremely ill conditioned, check evals and k(S)')
#         Eo = 0.0
#
#     if(return_all_eigs or return_S or return_Hbar):
#         return_list = [Eo]
#         if(return_all_eigs):
#             return_list.append(evals_sorted)
#         if(return_S):
#             return_list.append(s_mat)
#         if(return_Hbar):
#             return_list.append(h_mat)
#
#         return return_list
#
#     return Eo

def adaptive_rtlvqe_energy(mol, Nrefs, mr_dt, initial_ref, maxiter=100,
                            target_root=0, Ninitial_states=4, inital_dt=1.0,
                            fast=False, nstates_per_ref=2,
                            print_mats=True, return_all_eigs=False,
                            return_S=False, return_Hbar=False, optemizer = 'nelder-mead'):

    # Below instructions will be executed by a function that returns a vector of reffs.
    ref_lst = artl_helpers.get_init_ref_lst(initial_ref, Nrefs, Ninitial_states, inital_dt,
                                            mol, target_root=target_root, fast=True)


    myVQE = vqe.RTLVQE(ref_lst, nstates_per_ref, mol.get_hamiltonian(), fast=True,
                        N_samples = 100, optimizer=optemizer)
    myVQE.do_vqe(mr_dt, maxiter=maxiter)
    Energy = myVQE.get_energy()
    initial_Energy = myVQE.get_inital_guess_energy()

    return Energy, initial_Energy
