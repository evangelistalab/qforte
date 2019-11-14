import qforte
from qforte.utils import trotterization as trot
from qforte.rtl import rtl_helpers
from qforte.rtl.rtl_helpers import sorted_largest_idxs

import numpy as np
from scipy import linalg

import collections

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

def intiger_to_ref(n, nqubits):
    qb = qforte.QuantumBasis(n)
    ref = []
    for i in range(nqubits):
        if (qb.get_bit(i)):
            ref.append(1)
        else:
            ref.append(0)
    return ref

def open_shell(ref):
    norb = int(len(ref) / 2)
    for i in range(norb):
        i_alfa = 2*i
        i_beta = (2*i) + 1
        if((ref[i_alfa] + ref[i_beta]) == 1):
            return True

    return False

def correct_spin(ref, abs_spin):
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
    temp = ref.copy()
    i_alfa = 2*orb_idx
    i_beta = (2*orb_idx) + 1
    alfa_val = ref[i_alfa]
    beta_val = ref[i_beta]

    temp[i_alfa] = beta_val
    temp[i_beta] = alfa_val
    return temp

def build_eq_dets(open_shell_ref):
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

def get_init_ref_lst(initial_ref, d, Ninitial_states, inital_dt,
                    mol, target_root=None, fast=True,
                    use_phase_based_selection=False):

    initial_ref_lst = []
    true_initial_ref_lst = []

    #NOTE: need get nqubits from Molecule class attribute instead of initial_ref list length
    # Also true for UCC functions
    nqubits = len(initial_ref)

    h_mat = np.zeros((Ninitial_states,Ninitial_states), dtype=complex)
    s_mat = np.zeros((Ninitial_states,Ninitial_states), dtype=complex)

    Nis_untruncated = Ninitial_states

    if(fast):
        s_mat, h_mat = rtl_helpers.get_sr_mats_fast(initial_ref, inital_dt,
                                                    Ninitial_states, mol.get_hamiltonian(),
                                                    nqubits)

    else:
        for p in range(Ninitial_states):
            for q in range(p, Ninitial_states):
                h_mat[p][q] = rtl_helpers.matrix_element(initial_ref, inital_dt, p, q, mol.get_hamiltonian(),
                                                nqubits, mol.get_hamiltonian())
                h_mat[q][p] = np.conj(h_mat[p][q])

                s_mat[p][q] = rtl_helpers.matrix_element(initial_ref, inital_dt, p, q, mol.get_hamiltonian(),
                                                nqubits)
                s_mat[q][p] = np.conj(s_mat[p][q])

    print('\n\n    ==> Initial QK Matricies (for ref. selection)  <==')
    print('-----------------------------------------------------------')

    print("\nS initial:\n")
    rtl_helpers.matprint(s_mat)

    print("\nH initial:\n")
    rtl_helpers.matprint(h_mat)

    cs_str = '{:.2e}'.format(np.linalg.cond(s_mat))
    print('\nCondition number of overlap mat k(S):   ', cs_str)

    evals, evecs = rtl_helpers.canonical_geig_solve(s_mat, h_mat)

    if(Ninitial_states > len(evals)):
        print('\n', Ninitial_states, ' initial states requested, but QK produced ',
                    len(evals), ' stable roots.\n Using ', len(evals),
                    'intial states instead.')

        Ninitial_states = len(evals)

    sorted_evals_idxs = sorted_largest_idxs(evals, use_real=True, rev=False)
    sorted_evals = np.zeros((Ninitial_states), dtype=complex)
    sorted_evecs = np.zeros((Nis_untruncated,Ninitial_states), dtype=complex)
    for n in range(Ninitial_states):
        old_idx = sorted_evals_idxs[n][1]
        sorted_evals[n]   = evals[old_idx]
        sorted_evecs[:,n] = evecs[:,old_idx]

    sorted_sq_mod_evecs = np.zeros((Nis_untruncated,Ninitial_states), dtype=complex)

    for p in range(Nis_untruncated):
        for q in range(Ninitial_states):
            sorted_sq_mod_evecs[p][q] = sorted_evecs[p][q] * np.conj(sorted_evecs[p][q])

    basis_coeff_vec_lst = []
    for n in range(Nis_untruncated):
        if(fast):

            Uk = qforte.QuantumCircuit()
            for j in range(nqubits):
                if initial_ref[j] == 1:
                    Uk.add_gate(qforte.make_gate('X', j, j))

            temp_op1 = qforte.QuantumOperator()
            for t in mol.get_hamiltonian().terms():
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
    for n in range(Ninitial_states):
        for i, val in enumerate(Cprime[n]):
            Cprime[n][i] *= np.conj(val)

    for n in range(Ninitial_states):
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

def get_sa_init_ref_lst(initial_ref, d, Ninitial_states, inital_dt,
                    mol, target_root=None, fast=True,
                    use_phase_based_selection=False):

    if(fast==False):
        raise NotImplementedError('Only fast algorithm avalible for get_sa_init_ref_lst')


    ref_lst = get_init_ref_lst(initial_ref, 2*d, Ninitial_states, inital_dt,
                                        mol, target_root=target_root, fast=True,
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
    s_mat, h_mat = rtl_helpers.get_mr_mats_fast(pre_sa_ref_lst, 1,
                                                dt_lst, mol.get_hamiltonian(),
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
        qf_det_idx = rtl_helpers.ref_to_basis_idx(det)
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

            qf_det_idx = rtl_helpers.ref_to_basis_idx(temp[1])
            basis = qforte.QuantumBasis(qf_det_idx)
            if(temp[0] > 0.0):
                print('   ', round(temp[0], 4), '     ', basis.str(nqubits))
            else:
                print('  ', round(temp[0], 4), '     ', basis.str(nqubits))

        sa_ref_lst.append(basis_vec)

    return sa_ref_lst
