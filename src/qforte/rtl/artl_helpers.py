import qforte
# from qforte.utils import transforms
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

# NOTE: This function needs to be proofed to make sure correct dets are included
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

    print('\nbefore addtion:\n')
    print(eq_ref_lst1)

    for ref in eq_ref_lst1:
        for orb in one_e_orbs:
            temp = flip_spin(ref, orb)
            if(temp not in eq_ref_lst1):
                eq_ref_lst1.append(temp)

    print('\nall dets after addition:\n')
    print(eq_ref_lst1)

    eq_ref_lst2 = []
    for ref in eq_ref_lst1:
        if(correct_spin(ref, abs_spin)):
            eq_ref_lst2.append(ref)

    return eq_ref_lst2

def get_init_ref_lst(initial_ref, Nrefs, Ninitial_states, inital_dt,
                    mol, target_root=None, fast=True,
                    use_phase_based_selection=False):

    # if(use_phase_based_selection):
    #     raise NotImplementedError("Phase based selection not yet supported for get_init_ref_lst().")

    initial_ref_lst = []
    true_initial_ref_lst = []

    #NOTE: need get nqubits from Molecule class attribute instead of initial_ref list length
    # Also true for UCC functions
    nqubits = len(initial_ref)

    h_mat = np.zeros((Ninitial_states,Ninitial_states), dtype=complex)
    s_mat = np.zeros((Ninitial_states,Ninitial_states), dtype=complex)

    Nis_untruncated = Ninitial_states

    if(fast):
        print('using faster fast algorithm lol')
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

    print("\nS initial:\n")
    rtl_helpers.matprint(s_mat)

    print('cond(S):', np.linalg.cond(s_mat))

    print("\nHbar: initial\n")
    rtl_helpers.matprint(h_mat)

    evals, evecs = rtl_helpers.canonical_geig_solve(s_mat, h_mat)

    ####################################

    # evals, evecs = linalg.eig(h_mat,s_mat)

    if(Ninitial_states > len(evals)):
        print('\n', Ninitial_states, ' initial states requested, but SR-RTQL produced ',
                    len(evals), ' stable roots.\n Using ', len(evals),
                    'intial states instead.')

        Ninitial_states = len(evals)


    # need to make sorted evals and evecs...
    # use sorted_largest_idxs()
    sorted_evals_idxs = sorted_largest_idxs(evals, use_real=True, rev=False)
    sorted_evals = np.zeros((Ninitial_states), dtype=complex)
    sorted_evecs = np.zeros((Nis_untruncated,Ninitial_states), dtype=complex)
    for n in range(Ninitial_states):
        old_idx = sorted_evals_idxs[n][1]
        sorted_evals[n]   = evals[old_idx]
        sorted_evecs[:,n] = evecs[:,old_idx]

    print('\nARTLanczos (sorted!) initial evals:\n\n', sorted_evals)
    print('\nARTLanczos initial sorted evecs:\n')
    matprint(sorted_evecs)


    sorted_sq_mod_evecs = np.zeros((Nis_untruncated,Ninitial_states), dtype=complex)

    for p in range(Nis_untruncated):
        for q in range(Ninitial_states):
            sorted_sq_mod_evecs[p][q] = sorted_evecs[p][q] * np.conj(sorted_evecs[p][q])

    print('\nARTLanczos initial sorted evecs square modulous:\n')
    matprint(sorted_sq_mod_evecs)

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
            raise ValueError('Measurement-based selection of new refs not yet implemented.')

    # Actual Values
    basis_coeff_mat = np.array(basis_coeff_vec_lst)

    Cprime = (np.conj(sorted_evecs.transpose())).dot(basis_coeff_mat)
    print('\nshape Cprime: ', np.shape(Cprime))
    for n in range(Ninitial_states):
        for i, val in enumerate(Cprime[n]):
            Cprime[n][i] *= np.conj(val)

    # Now using the approximation
    for n in range(Ninitial_states):
        for i, val in enumerate(basis_coeff_vec_lst[n]):
            basis_coeff_vec_lst[n][i] *= np.conj(val)

    Cprime_sq_mod = (sorted_sq_mod_evecs.transpose()).dot(basis_coeff_mat)

    true_idx_lst = []
    idx_lst = []
    if(target_root is not None):
        print('\nTargeting refs for root ', target_root)

        # ###########
        #
        # print('approx ground state wvfn sq mod')
        # print(Cprime[target_root,:])
        #
        # ###########

        true_sorted_idxs = sorted_largest_idxs(Cprime[target_root,:])
        sorted_idxs = sorted_largest_idxs(Cprime_sq_mod[target_root,:])
        # idx_lst.append( initial_ref )
        for n in range(Nrefs):
            # if(sorted_idxs[n][1]!=0):
            idx_lst.append( sorted_idxs[n][1] )

            true_idx_lst.append( true_sorted_idxs[n][1] )

    else:
        raise ValueError("psudo-state avaraged selection approach not yet functional")
        if(Nrefs > Ninitial_states):
            raise ValueError("Can't get more refs than states for psudo-state avaraged approach")
        for n in range(Nrefs):
            true_sorted_idxs = sorted_largest_idxs(Cprime[target_root,:])
            sorted_idxs = sorted_largest_idxs(Cprime_sq_mod[n,:])
            if(sorted_idxs[0][1] in idx_lst):
                k = 1
                while(sorted_idxs[k][1] in idx_lst):
                    k += 1
                    if(k == 2**nqubits):
                        raise ValueError('Selection of inital determinant references unsucessful.')

                l = 1
                while(true_sorted_idxs[k][1] in true_idx_lst):
                    l += 1
                    if(l == 2**nqubits):
                        raise ValueError('Selection of inital determinant references unsucessful.')

                true_idx_lst.append( true_sorted_idxs[k][1] )
                idx_lst.append( sorted_idxs[k][1] )


            else:
                true_idx_lst.append( true_sorted_idxs[0][1] )
                idx_lst.append( sorted_idxs[0][1] )

    print('\nMost important determinats (no approximation):\n')
    print('root               dominant determinant  ')
    print('----------------------------------------')
    for i, idx in enumerate(true_idx_lst):
        basis = qforte.QuantumBasis(idx)
        print('  ', i, '                ', basis.str(nqubits))


    print('\nInitial ref guesses (using approximation):\n')
    print('root               dominant determinant  ')
    print('----------------------------------------')
    for i, idx in enumerate(idx_lst):
        basis = qforte.QuantumBasis(idx)
        print('  ', i, '                ', basis.str(nqubits))



    for idx in true_idx_lst:
        true_initial_ref_lst.append(intiger_to_ref(idx, nqubits))

    if(initial_ref not in true_initial_ref_lst):
        print('\nAdding initial referance determinant\n')
        for i in range(len(true_initial_ref_lst) - 1):
            true_initial_ref_lst[i+1] = true_initial_ref_lst[i]

        true_initial_ref_lst[0] = initial_ref



    for idx in idx_lst:
        initial_ref_lst.append(intiger_to_ref(idx, nqubits))

    ### ###
    # print('\ninital ref lst before addtion')
    # for ref in initial_ref_lst:
    #     print(' ', ref)
    ### ###

    if(initial_ref not in initial_ref_lst):
        print('\nAdding initial referance determinant\n')
        staggard_initial_ref_lst = [initial_ref]
        for i in range(len(initial_ref_lst) - 1):
            # initial_ref_lst[i+1] = initial_ref_lst[i].copy()
            staggard_initial_ref_lst.append(initial_ref_lst[i].copy())

        initial_ref_lst[0] = initial_ref
        initial_ref_lst = staggard_initial_ref_lst.copy()

    ### ###
    # print('\ninital ref lst AFTER addtion')
    # for ref in initial_ref_lst:
    #     print(' ', ref)
    ### ###

    if(use_phase_based_selection):
        print('\nusing phase based reference selection!\n')
        return true_initial_ref_lst

    else:
        return initial_ref_lst

# def get_adaptive_init_ref_lst(initial_ref, Nrefs, Ninitial_states, inital_dt,
#                     mol, refs_per_iter=5, target_root=None, fast=True,
#                     use_phase_based_selection=False):
#
#     initial_ref_lst = get_init_ref_lst(initial_ref, refs_per_iter, Ninitial_states, inital_dt,
#                         mol, target_root=target_root, fast=fast,
#                         use_phase_based_selection=use_phase_based_selection)
#
#
#
#         return initial_ref_lst


def get_sa_init_ref_lst(initial_ref, Nrefs, Ninitial_states, inital_dt,
                    mol, target_root=None, fast=True,
                    use_phase_based_selection=False):

    if(fast==False):
        raise NotImplementedError('Only fast algorithm avalible for get_sa_init_ref_lst')


    ref_lst = get_init_ref_lst(initial_ref, 2*Nrefs, Ninitial_states, inital_dt,
                                        mol, target_root=target_root, fast=True,
                                        use_phase_based_selection=use_phase_based_selection)

    # print('\ninitial reference list before spin adaptation\n')
    # print(ref_lst)


    print('\nspin adapting guess reference list\n')

    # first build ref list for consturction of Htilde
    pre_sa_ref_lst = []
    num_refs_per_config = []
    # Might also need to build list for indexes of eqivilant dets

    for ref in ref_lst:
        print('  ref:', ref)
        for other_ref in pre_sa_ref_lst:
            print('    ref_in_pre_sa_lst:', other_ref)
        if(ref not in pre_sa_ref_lst):
            if(open_shell(ref)):
                temp = build_eq_dets(ref)
                pre_sa_ref_lst = pre_sa_ref_lst + temp
                num_refs_per_config.append(len(temp))
            else:
                pre_sa_ref_lst.append(ref)
                num_refs_per_config.append(1)

    print('\nlen(pre_sa_ref_lst):', len(pre_sa_ref_lst))
    print('\nsum(num_refs_per_config):', sum(num_refs_per_config))

    print('\npre_sa_ref_lst:', pre_sa_ref_lst)
    print('\nnum_refs_per_config:', num_refs_per_config)
    # At this point pre_sa_ref_lst contains correct spin combos

    print('\n\npre_sa_ref list:')
    print('----------------------------')
    for ref in pre_sa_ref_lst:
        print(ref)
    print('')

    print('\n\nPossible repeats in pre_sa_ref_list!')
    _size = len(pre_sa_ref_lst)
    repeated = []
    for i in range(_size):
        k = i + 1
        for j in range(k, _size):
            if pre_sa_ref_lst[i] == pre_sa_ref_lst[j] and pre_sa_ref_lst[i] not in repeated:
                repeated.append(pre_sa_ref_lst[i])
    for ref in repeated:
        print(ref)

    # Now we need to create a CI with these dets
    # NOTE: again, stop getting the number of qubits this way...
    nqubits = len(pre_sa_ref_lst[0])
    dt_lst = np.zeros(len(pre_sa_ref_lst))
    s_mat, h_mat = rtl_helpers.get_mr_mats_fast(pre_sa_ref_lst, 1,
                                                dt_lst, mol.get_hamiltonian(),
                                                nqubits)

    print('\n\ncondition number of pure CI (sould be 1.0):')
    print(np.linalg.cond(s_mat))

    # print('\noverlap for pre spin adapted configuration determination:')
    # matprint(s_mat)

    # print('\nhamiltonian for pre spin adapted configuration determination:')
    # matprint(h_mat)

    evals, evecs = linalg.eig(h_mat)

    print('\nevals from pre_sa_list', evals)
    # print('\nevecs from pre_sa_list:')
    # matprint(evecs)

    #Sort Evals and Evecs
    sorted_evals_idxs = sorted_largest_idxs(evals, use_real=True, rev=False)
    sorted_evals = np.zeros((len(evals)), dtype=float)
    sorted_evecs = np.zeros(np.shape(evecs), dtype=float)
    for n in range(len(evals)):
        old_idx = sorted_evals_idxs[n][1]
        sorted_evals[n]   = np.real(evals[old_idx])
        sorted_evecs[:,n] = np.real(evecs[:,old_idx])

    print('\nsorted evals from pre_sa_list', sorted_evals)
    print('\nsorted evecs from pre_sa_list[root 0]: \n', sorted_evecs[:,0])

    print('\nsorted evecs from pre_sa_list[root 1]: \n', sorted_evecs[:,1])

    print('\nsorted evecs from pre_sa_list[root 2]: \n', sorted_evecs[:,2])

    print('\nsorted evecs from pre_sa_list[root 3]: \n', sorted_evecs[:,3])
    # matprint(sorted_evecs)

    # Make Coeff and determinats lists
    if(np.abs(sorted_evecs[:,0][0]) < 1.0e-6):
        print('Small CI ground state likely of wrong symmetry, trying other roots!')
        max = len(sorted_evals)
        adjusted_root = 0
        Co_val = 0.0
        while (Co_val < 1.0e-6):
            adjusted_root += 1
            Co_val = np.abs(sorted_evecs[:,adjusted_root][0])


        target_root = adjusted_root
        print('now using small CI root: ', target_root)

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
            # print('\nval: ', val, ' sum ', summ)
        temp = [x / np.sqrt(summ) for x in target_state[start:end]]
        norm_basis_coeff_lst.append(temp)

        basis_coeff_lst.append(target_state[start:end])
        det_lst.append(pre_sa_ref_lst[start:end])
        coeff_idx += num_refs

    print('\nbasis coefficient list:')
    for i, coeff in enumerate(basis_coeff_lst):
        print(norm_basis_coeff_lst[i])
        print(coeff)
        print(det_lst[i])

    # wil be used to sort the final sa_ref_lst and basis_coeff_lst
    basis_importnace_lst = []
    for basis_coeff in basis_coeff_lst:
        for coeff in basis_coeff:
            val = 0.0
            val += coeff*coeff
        basis_importnace_lst.append(val)

    print('\nbasis importance list:')
    print(basis_importnace_lst)
    sorted_basis_importnace_lst = sorted_largest_idxs(basis_importnace_lst, use_real=True, rev=True)

    # Construct final ref list, of form
    # [ [ (coeff, [1100]) ], [ (coeff, [1001]), (coeff, [0110]) ], ... ]
    sa_ref_lst = []
    for i in range(Nrefs):
        old_idx = sorted_basis_importnace_lst[i][1]
        basis_vec = []
        for k in range( len(basis_coeff_lst[old_idx]) ):
            temp = ( norm_basis_coeff_lst[old_idx][k], det_lst[old_idx][k] )
            basis_vec.append( temp )
        sa_ref_lst.append(basis_vec)

    # print('\n\nsa_ref_lst:')
    # print(sa_ref_lst)


    # For now just returning regular ref list

    # return pre_sa_ref_lst[0:Nrefs]
    return sa_ref_lst























    #
