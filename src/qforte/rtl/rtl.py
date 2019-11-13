import qforte
from qforte.rtl import rtl_helpers
# from qforte.utils import transforms
# from qforte.utils import trotterization as trot

import numpy as np
from scipy import linalg


    ####################################### READ ###########################################
    # This module requires you to pull the rtl_pilot branch of QForte, it contains         #
    # the 'perfect_measurement' function as well as the ability to exponeniate operators   #
    # as controlled-unitaries                                                              #
    ########################################################################################


def rtl_energy(mol, ref, dt, nstates, fast=False, print_mats=True, return_all_eigs=False, return_S=False, return_Hbar=False):

    #NOTE: need get nqubits from Molecule class attribute instead of ref list length
    # Also true for UCC functions
    nqubits = len(ref)

    h_mat = np.zeros((nstates,nstates), dtype=complex)
    s_mat = np.zeros((nstates,nstates), dtype=complex)

    if(fast):
        #print("\n      ====> Using fast version of algorithm. <====")
        s_mat, h_mat = rtl_helpers.get_sr_mats_fast(ref, dt,
                                                    nstates, mol.get_hamiltonian(),
                                                    nqubits)

    else:
        for p in range(nstates):
            for q in range(p, nstates):
                h_mat[p][q] = rtl_helpers.matrix_element(ref, dt, p, q, mol.get_hamiltonian(),
                                                nqubits, mol.get_hamiltonian())

                h_mat[q][p] = np.conj(h_mat[p][q])

                s_mat[p][q] = rtl_helpers.matrix_element(ref, dt, p, q, mol.get_hamiltonian(),
                                                nqubits)

                s_mat[q][p] = np.conj(s_mat[p][q])

    if(print_mats):
        print('------------------------------------------------')
        print('     Matricies for Quantum Real-Time Lanczos')
        print('------------------------------------------------')
        print('Nsteps  : ', nstates)
        print('delta t :     ', dt)
        print("\nS:\n")
        rtl_helpers.matprint(s_mat)

        print("\nHbar:\n")
        rtl_helpers.matprint(h_mat)

    evals, evecs = rtl_helpers.canonical_geig_solve(s_mat, h_mat)
    # evals, evecs = linalg.eig(h_mat,s_mat)

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
            return_list.append(evals_sorted)
        if(return_S):
            return_list.append(s_mat)
        if(return_Hbar):
            return_list.append(h_mat)

        return return_list

    return Eo


def mr_rtl_energy(mol, ref_lst, C_ref_lst, fast=False, const_dt=None, nstates_per_ref=1, print_mats=True, return_all_eigs=False, return_S=False, return_Hbar=False):

    # if(nstates_per_ref != 1):
    #     raise ValueError('Using multiple time evolutions for each reference determinant is not yet supported.')

    if(len(ref_lst) != len(C_ref_lst)):
        raise ValueError('need same number of C guesses as references provided.')

    #NOTE: need get nqubits from Molecule class attribute instead of ref list length
    # Also true for UCC functions
    num_refs = len(ref_lst)
    num_tot_basis = num_refs * nstates_per_ref

    nqubits = len(ref_lst[0])

    h_mat = np.zeros((num_tot_basis,num_tot_basis), dtype=complex)
    s_mat = np.zeros((num_tot_basis,num_tot_basis), dtype=complex)

    # Build list of dt values to be used
    dt_lst = []

    if(const_dt):
        for i in range(len(ref_lst)):
            dt_lst.append(const_dt)

    else:
        for C in C_ref_lst:
            dt_lst.append( (2*np.pi * (1.0 - C*np.conj(C))) )

    # print('dt_lst: ', dt_lst)

    if(fast):
        # Fast algorimth doesn't use measurement
        for I in range(num_refs):
            for J in range(num_refs):
                for m in range(nstates_per_ref):
                    for n in range(nstates_per_ref):

                        p = I*nstates_per_ref + m
                        q = J*nstates_per_ref + n

                        if(q>=p):

                            # print('I: ', I)
                            # print(' J: ', J)
                            # print('  m: ', m)
                            # print('   n: ', n)
                            # print('    p: ', p, ' q: ', q)


                            ref_I = ref_lst[I]
                            ref_J = ref_lst[J]

                            dt_I = dt_lst[I]
                            dt_J = dt_lst[J]
                            # print('    ref_I:',  ref_I, '  ref_J: ', ref_J)
                            # print('     dt_I: ', dt_I,  '   dt_J: ', dt_J)


                            h_mat[p][q] = rtl_helpers.mr_matrix_element_fast(ref_I, ref_J, dt_I, dt_J,
                                                                        m, n, mol.get_hamiltonian(),
                                                                        nqubits, mol.get_hamiltonian())

                            h_mat[q][p] = np.conj(h_mat[p][q])

                            s_mat[p][q] = rtl_helpers.mr_matrix_element_fast(ref_I, ref_J, dt_I, dt_J,
                                                                        m, n, mol.get_hamiltonian(),
                                                                        nqubits)

                            s_mat[q][p] = np.conj(s_mat[p][q])

    else:
        # Most basic to ensure things are working
        for I in range(num_refs):
            for J in range(num_refs):
                for m in range(nstates_per_ref):
                    for n in range(nstates_per_ref):

                        p = I*nstates_per_ref + m
                        q = J*nstates_per_ref + n

                        if(q>=p):

                            # print('I: ', I)
                            # print(' J: ', J)
                            # print('  m: ', m)
                            # print('   n: ', n)
                            # print('    p: ', p, ' q: ', q)


                            ref_I = ref_lst[I]
                            ref_J = ref_lst[J]

                            dt_I = dt_lst[I]
                            dt_J = dt_lst[J]
                            # print('    ref_I:',  ref_I, '  ref_J: ', ref_J)
                            # print('     dt_I: ', dt_I,  '   dt_J: ', dt_J)


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


    evals, evecs = linalg.eig(h_mat,s_mat)

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
            return_list.append(evals_sorted)
        if(return_S):
            return_list.append(s_mat)
        if(return_Hbar):
            return_list.append(h_mat)

        return return_list

    return Eo
