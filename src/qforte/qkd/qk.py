import qforte
from qforte.qkd import qk_helpers

import numpy as np
from scipy import linalg


    ####################################### READ ###########################################
    # This module requires you to pull the rtl_pilot branch of QForte, it contains         #
    # the 'perfect_measurement' function as well as the ability to exponeniate operators   #
    # as controlled-unitaries                                                              #
    ########################################################################################


def qk_energy(mol, ref, dt, s,
                fast=False,
                trot_number = 1,
                print_mats=True,
                return_all_eigs=False,
                return_S=False,
                return_Hbar=False):

    #NOTE: need get nqubits from Molecule class attribute instead of ref list length
    # Also true for UCC functions

    print('\n-----------------------------------------------------')
    print('           Single Reference Quantum Krylov   ')
    print('-----------------------------------------------------')

    nqubits = len(ref)
    nstates = s+1
    init_basis_idx = qk_helpers.ref_to_basis_idx(ref)
    init_basis = qforte.QuantumBasis(init_basis_idx)

    print('\n\n                     ==> QK options <==')
    print('-----------------------------------------------------------')
    print('Reference:                               ',  init_basis.str(nqubits))
    print('Time evolutions per reference (s):       ',  s)
    print('Dimension of Krylov space (N):           ',  nstates)
    print('Delta t (in a.u.):                       ',  dt)
    print('Trotter number (m):                      ',  trot_number)
    print('Use fast version of algorithm:           ',  str(fast))

    h_mat = np.zeros((nstates,nstates), dtype=complex)
    s_mat = np.zeros((nstates,nstates), dtype=complex)

    if(fast):
        s_mat, h_mat = qk_helpers.get_sr_mats_fast(ref, dt,
                                                    nstates, mol.get_hamiltonian(),
                                                    nqubits, trot_number=trot_number)

    else:
        for p in range(nstates):
            for q in range(p, nstates):
                h_mat[p][q] = qk_helpers.matrix_element(ref, dt, p, q, mol.get_hamiltonian(),
                                                nqubits, mol.get_hamiltonian(), trot_number=trot_number)

                h_mat[q][p] = np.conj(h_mat[p][q])

                s_mat[p][q] = qk_helpers.matrix_element(ref, dt, p, q, mol.get_hamiltonian(),
                                                nqubits, trot_number=trot_number)

                s_mat[q][p] = np.conj(s_mat[p][q])

    if(print_mats):
        print('------------------------------------------------')
        print('         Matricies for Quantum Krylov')
        print('------------------------------------------------')

        print("\nS:\n")
        qk_helpers.matprint(s_mat)

        print("\nH:\n")
        qk_helpers.matprint(h_mat)

    evals, evecs = qk_helpers.canonical_geig_solve(s_mat, h_mat)

    evals_sorted = np.sort(evals)

    if(np.abs(np.imag(evals_sorted[0])) < 1.0e-3):
        Eo = np.real(evals_sorted[0])
    elif(np.abs(np.imag(evals_sorted[1])) < 1.0e-3):
        print('Warning: problem may be ill condidtiond, evals have imaginary components')
        Eo = np.real(evals_sorted[1])
    else:
        print('Warding: problem may be extremely ill conditioned, check evals and k(S)')
        Eo = 0.0

    print('\n\n                     ==> QK summary <==')
    print('-----------------------------------------------------------')
    cs_str = '{:.2e}'.format(np.linalg.cond(s_mat))
    print('Condition number of overlap mat k(S):     ', cs_str)
    print('Final QK Energy:                          ', round(Eo, 10))

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
