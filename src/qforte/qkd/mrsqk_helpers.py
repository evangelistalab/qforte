"""
qk_helpers.py
=================================================
A module containing helper functions for
multireference features of quantum Krylov
algorithms.
"""

import qforte
from qforte.utils import trotterization as trot
from qforte.qkd import qk_helpers
from qforte.qkd.qk_helpers import sorted_largest_idxs

import numpy as np
from scipy import linalg

import collections

def matprint(mat, fmt="g"):
    """Prints (2 X 2) numpy arrays in an intelligable fashion.

        Arguments
        ---------

        mat : ndarray
            A real (or complex) 2 X 2 numpt array to be printed.

    """
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

def intiger_to_ref(n, nqubits):
    """Takes an intager pertaining to a biary number and returns the corresponding
    determinant occupation list (reference).

        Arguments
        ---------

        n : int
            The index.

        nqubits : int
            The number of qubits (will be length of list).

        Returns
        -------

        ref : list
            A list of 1's and 0's representing spin orbital occupations for a
            single Slater determinant.

    """
    qb = qforte.QuantumBasis(n)
    ref = []
    for i in range(nqubits):
        if (qb.get_bit(i)):
            ref.append(1)
        else:
            ref.append(0)
    return ref

def open_shell(ref):
    """Determines Wheter or not the reference is an open shell determinant.
    Returns True if ref is open shell and False if not.

        Arguments
        ---------

        ref : list
            A list of 1's and 0's representing spin orbital occupations for a
            single Slater determinant.

    """
    norb = int(len(ref) / 2)
    for i in range(norb):
        i_alfa = 2*i
        i_beta = (2*i) + 1
        if((ref[i_alfa] + ref[i_beta]) == 1):
            return True

    return False

def correct_spin(ref, abs_spin):
    """Determines Wheter or not the reference has correct spin.
    Returns True if ref is ref spin matches overall spin and False if not.

        Arguments
        ---------

        ref : list
            A list of 1's and 0's representing spin orbital occupations for a
            single Slater determinant.

        abs_spin : float
            The targeted spin value.

    """
    if (abs_spin != 0):
        raise NotImplementedError("MRSQK currently only supports singlet state calculations.")

    norb = int(len(ref) / 2)
    spin = 0.0
    for i in range(norb):
        i_alfa = 2*i
        i_beta = (2*i) + 1
        spin += ref[i_alfa] * 0.5
        spin -= ref[i_beta] * 0.5

    if(np.abs(spin) == abs_spin):
        return True
    else:
        return False

def flip_spin(ref, orb_idx):
    """Takes in a single determinant reference and returns a determinant with
    the spin of the spin of the specified spatial orbtital (orb_idx) flipped.
    If the specified spatail orbital is doubly occupied, then the same
    determinant is returned.

        Arguments
        ---------

        ref : list
            A list of 1's and 0's representing spin orbital occupations for a
            single Slater determinant.

        orb_idx : int
            An index for the spatial orbtial of interest.

        Retruns
        -------

        temp : list
            A list of 1's and 0's representing spin orbital occupations for a
            single Slater determinant, with the spin of the specified spatial
            orbital flipped.

    """
    temp = ref.copy()
    i_alfa = 2*orb_idx
    i_beta = (2*orb_idx) + 1
    alfa_val = ref[i_alfa]
    beta_val = ref[i_beta]

    temp[i_alfa] = beta_val
    temp[i_beta] = alfa_val
    return temp

def build_eq_dets(open_shell_ref):
    """Builds a list of unique spin equivalent determinants from an open shell
    determinant. For example, if [1,0,0,1] is given as in input, it will return
    [[1,0,0,1], [0,1,1,0]].

        Arguments
        ---------

        open_shell_ref : list
            A list of 1's and 0's representing spin orbital occupations for a
            single open-shell Slater determinant.

        Returns
        -------

        eq_ref_lst2 : list of lists
            A list of open-shell determinants which are spin equivalent to
            open_shell_ref (including open_shell_ref).

    """
    norb = int(len(open_shell_ref) / 2)
    one_e_orbs = []
    spin = 0.0
    for i in range(norb):
        i_alfa = 2*i
        i_beta = (2*i) + 1
        spin += open_shell_ref[i_alfa] * 0.5
        spin -= open_shell_ref[i_beta] * 0.5

        if((open_shell_ref[i_alfa] + open_shell_ref[i_beta]) == 1):
            one_e_orbs.append(i)

    abs_spin = np.abs(spin)
    eq_ref_lst1 = [open_shell_ref]

    for ref in eq_ref_lst1:
        for orb in one_e_orbs:
            temp = flip_spin(ref, orb)
            if(temp not in eq_ref_lst1):
                eq_ref_lst1.append(temp)

    eq_ref_lst2 = []
    for ref in eq_ref_lst1:
        if(correct_spin(ref, abs_spin)):
            eq_ref_lst2.append(ref)

    return eq_ref_lst2

def get_init_ref_lst(initial_ref, d, ninitial_states, inital_dt,
                    H, target_root=0, fast=True,
                    use_phase_based_selection=False):
    """Builds a list of single determinant references to be used in the MRSQK
    procedure.

        Arguments
        ---------

        initial_ref : list
            A list of 1's and 0's representing spin orbital occupations for a
            single open-shell Slater determinant. It serves as the reference
            for the initial quantum Krylov calculation used to pick additional
            references.

        d : int
            The totoal number of references to find.

        ninitial_states : int
            The size of the inital quantum Krylov basis to use. Equal to the
            number of inital time evolutions plus one (s_0 +1).

        initial_dt : float
            The time step (delta t) to use in the inital quantum Krylov
            calculation.

        H : Molecule
            The Molecule object to use in MRSQK.

        target_root : int
            Determines which state to return the energy for.

        fast : bool
            Whether or not to use a faster version of the algorithm that bypasses
            measurment (unphysical for quantum computer).

        use_phase_based_selection : bool
            Whether or not to account for sign discrepencaies when selecting important
            determinants from initial quantum Krylov procedure.

        Returns
        -------

        initial_ref_lst : list of lists
            A list of the most important single determinants according to the initial
            quantum Krylov procedure.

    """

    initial_ref_lst = []
    true_initial_ref_lst = []

    #NOTE: need get nqubits from Molecule class attribute instead of initial_ref list length
    # Also true for UCC functions
    nqubits = len(initial_ref)

    h_mat = np.zeros((ninitial_states,ninitial_states), dtype=complex)
    s_mat = np.zeros((ninitial_states,ninitial_states), dtype=complex)

    Nis_untruncated = ninitial_states

    if(fast):
        s_mat, h_mat = qk_helpers.get_sr_mats_fast(initial_ref, inital_dt,
                                                    ninitial_states, H,
                                                    nqubits)

    else:
        for p in range(ninitial_states):
            for q in range(p, ninitial_states):
                h_mat[p][q] = qk_helpers.matrix_element(initial_ref, inital_dt, p, q, H,
                                                nqubits, H)
                h_mat[q][p] = np.conj(h_mat[p][q])

                s_mat[p][q] = qk_helpers.matrix_element(initial_ref, inital_dt, p, q, H,
                                                nqubits)
                s_mat[q][p] = np.conj(s_mat[p][q])

    print('\n\n    ==> Initial QK Matricies (for ref. selection)  <==')
    print('-----------------------------------------------------------')

    print("\nS initial:\n")
    qk_helpers.matprint(s_mat)

    print("\nH initial:\n")
    qk_helpers.matprint(h_mat)

    cs_str = '{:.2e}'.format(np.linalg.cond(s_mat))
    print('\nCondition number of overlap mat k(S):   ', cs_str)

    evals, evecs = qk_helpers.canonical_geig_solve(s_mat, h_mat)

    if(ninitial_states > len(evals)):
        print('\n', ninitial_states, ' initial states requested, but QK produced ',
                    len(evals), ' stable roots.\n Using ', len(evals),
                    'intial states instead.')

        ninitial_states = len(evals)

    sorted_evals_idxs = sorted_largest_idxs(evals, use_real=True, rev=False)
    sorted_evals = np.zeros((ninitial_states), dtype=complex)
    sorted_evecs = np.zeros((Nis_untruncated,ninitial_states), dtype=complex)
    for n in range(ninitial_states):
        old_idx = sorted_evals_idxs[n][1]
        sorted_evals[n]   = evals[old_idx]
        sorted_evecs[:,n] = evecs[:,old_idx]

    sorted_sq_mod_evecs = np.zeros((Nis_untruncated,ninitial_states), dtype=complex)

    for p in range(Nis_untruncated):
        for q in range(ninitial_states):
            sorted_sq_mod_evecs[p][q] = sorted_evecs[p][q] * np.conj(sorted_evecs[p][q])

    basis_coeff_vec_lst = []
    for n in range(Nis_untruncated):
        if(fast):

            Uk = qforte.QuantumCircuit()
            for j in range(nqubits):
                if initial_ref[j] == 1:
                    Uk.add_gate(qforte.make_gate('X', j, j))

            temp_op1 = qforte.QuantumOperator()
            for t in H.terms():
                c, op = t
                phase = -1.0j * n * inital_dt * c
                temp_op1.add_term(phase, op)

            expn_op1, phase1 = qforte.trotterization.trotterize(temp_op1)
            for gate in expn_op1.gates():
                Uk.add_gate(gate)

            qc = qforte.QuantumComputer(nqubits)
            qc.apply_circuit(Uk)
            basis_coeff_vec_lst.append(qc.get_coeff_vec())

        else:
            raise NotImplementedError('Measurement-based selection of new refs not yet implemented.')

    basis_coeff_mat = np.array(basis_coeff_vec_lst)

    Cprime = (np.conj(sorted_evecs.transpose())).dot(basis_coeff_mat)
    for n in range(ninitial_states):
        for i, val in enumerate(Cprime[n]):
            Cprime[n][i] *= np.conj(val)

    for n in range(ninitial_states):
        for i, val in enumerate(basis_coeff_vec_lst[n]):
            basis_coeff_vec_lst[n][i] *= np.conj(val)

    Cprime_sq_mod = (sorted_sq_mod_evecs.transpose()).dot(basis_coeff_mat)

    true_idx_lst = []
    idx_lst = []
    if(target_root is not None):

        true_sorted_idxs = sorted_largest_idxs(Cprime[target_root,:])
        sorted_idxs = sorted_largest_idxs(Cprime_sq_mod[target_root,:])
        for n in range(d):
            idx_lst.append( sorted_idxs[n][1] )
            true_idx_lst.append( true_sorted_idxs[n][1] )

    else:
        raise NotImplementedError("psudo-state avaraged selection approach not yet functional")

    print('\n\n      ==> Initial QK Determinat selection summary  <==')
    print('-----------------------------------------------------------')

    if(use_phase_based_selection):
        print('\nMost important determinats:\n')
        print('index                     determinant  ')
        print('----------------------------------------')
        for i, idx in enumerate(true_idx_lst):
            basis = qforte.QuantumBasis(idx)
            print('  ', i+1, '                ', basis.str(nqubits))

    else:
        print('\nMost important determinats:\n')
        print('index                     determinant  ')
        print('----------------------------------------')
        for i, idx in enumerate(idx_lst):
            basis = qforte.QuantumBasis(idx)
            print('  ', i+1, '                ', basis.str(nqubits))

    for idx in true_idx_lst:
        true_initial_ref_lst.append(intiger_to_ref(idx, nqubits))

    if(initial_ref not in true_initial_ref_lst):
        print('\n***Adding initial referance determinant!***\n')
        for i in range(len(true_initial_ref_lst) - 1):
            true_initial_ref_lst[i+1] = true_initial_ref_lst[i]

        true_initial_ref_lst[0] = initial_ref

    for idx in idx_lst:
        initial_ref_lst.append(intiger_to_ref(idx, nqubits))

    if(initial_ref not in initial_ref_lst):
        print('\n***Adding initial referance determinant!***\n')
        staggard_initial_ref_lst = [initial_ref]
        for i in range(len(initial_ref_lst) - 1):
            staggard_initial_ref_lst.append(initial_ref_lst[i].copy())

        initial_ref_lst[0] = initial_ref
        initial_ref_lst = staggard_initial_ref_lst.copy()

    if(use_phase_based_selection):
        return true_initial_ref_lst

    else:
        return initial_ref_lst

def get_sa_init_ref_lst(initial_ref, d, ninitial_states, inital_dt,
                    H, target_root=0, fast=True,
                    use_phase_based_selection=False):
    """Builds a list of spin adapted references to be used in the MRSQK procedure.

        Arguments
        ---------

        initial_ref : list
            A list of 1's and 0's representing spin orbital occupations for a
            single open-shell Slater determinant. It serves as the reference
            for the initial quantum Krylov calculation used to pick additional
            references.

        d : int
            The totoal number of references to find.

        ninitial_states : int
            The size of the inital quantum Krylov basis to use. Equal to the
            number of inital time evolutions plus one (s_0 +1).

        initial_dt : float
            The time step (delta t) to use in the inital quantum Krylov
            calculation.

        H : Molecule
            The Molecule object to use in MRSQK.

        target_root : int
            Determines which state to return the energy for.

        fast : bool
            Whether or not to use a faster version of the algorithm that bypasses
            measurment (unphysical for quantum computer).

        use_phase_based_selection : bool
            Whether or not to account for sign discrepencaies when selecting important
            determinants from initial quantum Krylov procedure.

        Returns
        -------

        sa_ref_lst : list of lists
            A list containing all of the spin adapted references selected in the
            initial quantum Krylov procedure.
            It is specifically a list of lists of pairs containing coefficient vales
            and a lists pertaning to single determinants.
            As an example,
            ref_lst = [ [ (1.0, [1,1,0,0]) ], [ (0.7071, [0,1,1,0]), (0.7071, [1,0,0,1]) ] ].

    """

    if(fast==False):
        raise NotImplementedError('Only fast algorithm avalible for get_sa_init_ref_lst')


    ref_lst = get_init_ref_lst(initial_ref, 2*d, ninitial_states, inital_dt,
                                        H, target_root=target_root, fast=True,
                                        use_phase_based_selection=use_phase_based_selection)

    pre_sa_ref_lst = []
    num_refs_per_config = []

    for ref in ref_lst:
        if(ref not in pre_sa_ref_lst):
            if(open_shell(ref)):
                temp = build_eq_dets(ref)
                pre_sa_ref_lst = pre_sa_ref_lst + temp
                num_refs_per_config.append(len(temp))
            else:
                pre_sa_ref_lst.append(ref)
                num_refs_per_config.append(1)

    nqubits = len(pre_sa_ref_lst[0])
    dt_lst = np.zeros(len(pre_sa_ref_lst))
    s_mat, h_mat = qk_helpers.get_mr_mats_fast(pre_sa_ref_lst, 1,
                                                dt_lst, H,
                                                nqubits)

    evals, evecs = linalg.eig(h_mat)

    sorted_evals_idxs = sorted_largest_idxs(evals, use_real=True, rev=False)
    sorted_evals = np.zeros((len(evals)), dtype=float)
    sorted_evecs = np.zeros(np.shape(evecs), dtype=float)
    for n in range(len(evals)):
        old_idx = sorted_evals_idxs[n][1]
        sorted_evals[n]   = np.real(evals[old_idx])
        sorted_evecs[:,n] = np.real(evecs[:,old_idx])

    if(np.abs(sorted_evecs[:,0][0]) < 1.0e-6):
        print('Small CI ground state likely of wrong symmetry, trying other roots!')
        max = len(sorted_evals)
        adjusted_root = 0
        Co_val = 0.0
        while (Co_val < 1.0e-6):
            adjusted_root += 1
            Co_val = np.abs(sorted_evecs[:,adjusted_root][0])

        target_root = adjusted_root
        print('Now using small CI root: ', target_root)

    target_state = sorted_evecs[:,target_root]
    basis_coeff_lst = []
    norm_basis_coeff_lst = []
    det_lst = []
    coeff_idx = 0
    for num_refs in num_refs_per_config:
        start = coeff_idx
        end = coeff_idx + num_refs

        summ = 0.0
        for val in target_state[start:end]:
            summ += val * val
        temp = [x / np.sqrt(summ) for x in target_state[start:end]]
        norm_basis_coeff_lst.append(temp)

        basis_coeff_lst.append(target_state[start:end])
        det_lst.append(pre_sa_ref_lst[start:end])
        coeff_idx += num_refs

    print('\n\n      ==> Small CI with spin adapted dets summary <==')
    print('-----------------------------------------------------------')
    print('\nList augmented to included all spin \nconfigurations for open shells.')
    print('\n  Coeff                    determinant  ')
    print('----------------------------------------')
    for i, det in enumerate(pre_sa_ref_lst):
        qf_det_idx = qk_helpers.ref_to_basis_idx(det)
        basis = qforte.QuantumBasis(qf_det_idx)
        if(target_state[i] > 0.0):
            print('   ', round(target_state[i], 4), '                ', basis.str(nqubits))
        else:
            print('  ', round(target_state[i], 4), '                ', basis.str(nqubits))

    basis_importnace_lst = []
    for basis_coeff in basis_coeff_lst:
        for coeff in basis_coeff:
            val = 0.0
            val += coeff*coeff
        basis_importnace_lst.append(val)

    sorted_basis_importnace_lst = sorted_largest_idxs(basis_importnace_lst, use_real=True, rev=True)

    # Construct final ref list, of form
    # [ [ (coeff, [1100]) ], [ (coeff, [1001]), (coeff, [0110]) ], ... ]
    print('\n\n        ==> Final MRSQK reference space summary <==')
    print('-----------------------------------------------------------')

    sa_ref_lst = []
    for i in range(d):
        print('\nRef ', i+1)
        print('---------------------------')
        old_idx = sorted_basis_importnace_lst[i][1]
        basis_vec = []
        for k in range( len(basis_coeff_lst[old_idx]) ):
            temp = ( norm_basis_coeff_lst[old_idx][k], det_lst[old_idx][k] )
            basis_vec.append( temp )

            qf_det_idx = qk_helpers.ref_to_basis_idx(temp[1])
            basis = qforte.QuantumBasis(qf_det_idx)
            if(temp[0] > 0.0):
                print('   ', round(temp[0], 4), '     ', basis.str(nqubits))
            else:
                print('  ', round(temp[0], 4), '     ', basis.str(nqubits))

        sa_ref_lst.append(basis_vec)

    return sa_ref_lst
