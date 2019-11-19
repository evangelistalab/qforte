"""
mrsqk.py
=================================================
A module for calculating the energies of quantum-
mechanical systems the multireference selected
quantum Krylov algorithm.
"""
import qforte
from qforte.qkd import qk_helpers
from qforte.qkd import mrsqk_helpers
import numpy as np
from scipy import linalg


    ####################################### READ ###########################################
    # This module requires you to pull the mrsqk_pilot branch of QForte, it contains       #
    # the 'perfect_measurement' function as well as the ability to exponeniate operators   #
    # as controlled-unitaries                                                              #
    ########################################################################################

def mrsqk_energy(mol, d, s, mr_dt, initial_ref,
                        fast=False,
                        trot_number = 1,
                        target_root=0,
                        use_phase_based_selection=False,
                        use_spin_adapted_refs=True,
                        s_0=4,
                        inital_dt=0.25,
                        print_mats=False,
                        return_all_eigs=False,
                        return_S=False,
                        return_Hbar=False):

    """Executes the MRSQK algorithm and generates the energy.

        Arguments
        ---------
        mol : Molecule
            The Molecule object to use in MRSQK.

        d : int
            The dimenion of the reference space (number of references) to be used.

        s : int
            The number of time evolutions to perform on each reference.

        mr_dt : float
            The time step (delta t) to use for the evolutions of each reference.

        initial_ref : list
            The initial reference state given as a list of 1's and 0's
            (e.g. the Hartree-Fock state).

        fast : bool
            Whether or not to use a faster version of the algorithm that bypasses
            measurment (unphysical for quantum computer).

        trot_number : int
            The Trotter number for the calculation
            (exact in the infinte limit)

        target_root : int
            Determines which state to return the energy for.

        use_phase_based_selection : bool
            Whether or not to account for sign discrepencaies when selecting important
            determinants from initial QK procedure.

        use_spin_adapted_refs : bool
            Whether or not to spin adapt selected open shell determinants into a
            single reference.

        s_0 : int
            The number of evolutions to perform in the initial (single reference)
            QK calculation used to determine important references.

        inital_dt : float
            The time step (delta t) to use in the initial (single reference)
            QK calculation used to determine important references.

        print_mats : bool
            Whether or not to print the MRSQK H and S matricies.

        return_all_eigs : bool
            Additionally retrun a list of all other
            root energies.

        return_S : ndarray
            Additionally return a ndarray containing the overlap matrix S used in
            MRSQK.

        return_Hbar : ndarray
            Additionally return a ndarray containing Hamiltonian H used in
            MRSQK.

        Retruns
        -------
        Eo : float
            The energy of the specified root given by MRSQK.

    """

    nstates_per_ref = s + 1
    ninitial_states = s_0 + 1

    print('\n-----------------------------------------------------')
    print('        Multreference Selected Quantum Krylov   ')
    print('-----------------------------------------------------')

    nqubits = len(initial_ref)
    init_basis_idx = qk_helpers.ref_to_basis_idx(initial_ref)
    init_basis = qforte.QuantumBasis(init_basis_idx)

    print('\n\n                   ==> MRSQK options <==')
    print('-----------------------------------------------------------')
    print('Initial reference:                       ',  init_basis.str(nqubits))
    print('Dimension of reference space (d):        ',  d)
    print('Time evolutions per reference (s):       ',  s)
    print('Dimension of Krylov space (N):           ',  d*nstates_per_ref)
    print('Delta t (in a.u.):                       ',  mr_dt)
    print('Trotter number (m):                      ',  trot_number)
    print('Target root:                             ',  str(target_root))
    print('Use det. selection with sign:            ',  str(use_phase_based_selection))
    print('Use spin adapted references:             ',  str(use_spin_adapted_refs))
    print('Use fast version of algorithm:           ',  str(fast))

    print('\n\n     ==> Initial QK options (for ref. selection)  <==')
    print('-----------------------------------------------------------')
    print('Number of initial time evolutions (s_o): ',  s_0)
    print('Dimension of inital Krylov space (N_o):  ',  ninitial_states)
    print('Initial delta t_o (in a.u.):             ',  inital_dt)


    if(use_spin_adapted_refs):
        sa_ref_lst = mrsqk_helpers.get_sa_init_ref_lst(initial_ref, d, ninitial_states, inital_dt,
                                           mol.get_hamiltonian(), target_root=target_root, fast=True,
                                           use_phase_based_selection=use_phase_based_selection)

        nqubits = len(sa_ref_lst[0][0][1])

    else:
        ref_lst = mrsqk_helpers.get_init_ref_lst(initial_ref, d, ninitial_states, inital_dt,
                                            mol.get_hamiltonian(), target_root=target_root, fast=True,
                                            use_phase_based_selection=use_phase_based_selection)

        nqubits = len(ref_lst[0])

    #NOTE: need get nqubits from Molecule class attribute instead of ref list length
    # Also true for UCC functions
    num_refs = d
    num_tot_basis = num_refs * nstates_per_ref

    h_mat = np.zeros((num_tot_basis,num_tot_basis), dtype=complex)
    s_mat = np.zeros((num_tot_basis,num_tot_basis), dtype=complex)

    dt_lst = []
    for i in range(d):
        dt_lst.append(mr_dt)

    if(fast):
        if(use_spin_adapted_refs):
            s_mat, h_mat = qk_helpers.get_sa_mr_mats_fast(sa_ref_lst, nstates_per_ref,
                                                        dt_lst, mol.get_hamiltonian(),
                                                        nqubits, trot_number=trot_number)

        else:
            s_mat, h_mat = qk_helpers.get_mr_mats_fast(ref_lst, nstates_per_ref,
                                                        dt_lst, mol.get_hamiltonian(),
                                                        nqubits, trot_number=trot_number)

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

                            h_mat[p][q] = qk_helpers.mr_matrix_element(ref_I, ref_J, dt_I, dt_J,
                                                                        m, n, mol.get_hamiltonian(),
                                                                        nqubits, mol.get_hamiltonian(),
                                                                        trot_number=trot_number)
                            h_mat[q][p] = np.conj(h_mat[p][q])

                            s_mat[p][q] = qk_helpers.mr_matrix_element(ref_I, ref_J, dt_I, dt_J,
                                                                        m, n, mol.get_hamiltonian(),
                                                                        nqubits,
                                                                        trot_number=trot_number)
                            s_mat[q][p] = np.conj(s_mat[p][q])



    if(print_mats):
        print('\n\n                ==> MRSQK matricies <==')
        print('-----------------------------------------------------------')

        print("\nS:\n")
        qk_helpers.matprint(s_mat)
        print('\nk(S): ', np.linalg.cond(s_mat))

        print("\nHbar:\n")
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
