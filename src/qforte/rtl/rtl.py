import qforte
from qforte.rtl import rtl_helpers

import numpy as np
from scipy import linalg


    ####################################### READ ###########################################
    # This module requires you to pull the rtl_pilot branch of QForte, it contains         #
    # the 'perfect_measurement' function as well as the ability to exponeniate operators   #
    # as controlled-unitaries                                                              #
    ########################################################################################


def rtl_energy(mol, ref, dt, s, fast=False, print_mats=True, return_all_eigs=False, return_S=False, return_Hbar=False):

    #NOTE: need get nqubits from Molecule class attribute instead of ref list length
    # Also true for UCC functions
    nqubits = len(ref)
    nstates = s+1

    h_mat = np.zeros((nstates,nstates), dtype=complex)
    s_mat = np.zeros((nstates,nstates), dtype=complex)

    if(fast):
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
