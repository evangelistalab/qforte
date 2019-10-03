import qforte
# from qforte.utils import transforms
from qforte.utils import trotterization as trot
from qforte.rtl import rtl_helpers

import numpy as np
from scipy import linalg

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")


def sorted_largest_idxs(array, use_real=False, rev=True):
        temp = np.empty((len(array)), dtype=object )
        for i, val in enumerate(array):
            temp[i] = (val, i)
        if(use_real):
            sorted_temp = sorted(temp, key=lambda factor: np.real(factor[0]), reverse=rev)
        else:
            sorted_temp = sorted(temp, key=lambda factor: factor[0], reverse=rev)
        return sorted_temp

def intiger_to_ref(n, nqubits):
    qb = qforte.QuantumBasis(n)
    ref = []
    for i in range(nqubits):
        if (qb.get_bit(i)):
            ref.append(1)
        else:
            ref.append(0)
    return ref

def get_init_ref_lst(initial_ref, Ninitial_refs, inital_dt,
                    mol, target_root=None, fast=True):

    initial_ref_lst = []

    #NOTE: need get nqubits from Molecule class attribute instead of initial_ref list length
    # Also true for UCC functions
    nqubits = len(initial_ref)

    h_mat = np.zeros((Ninitial_refs,Ninitial_refs), dtype=complex)
    s_mat = np.zeros((Ninitial_refs,Ninitial_refs), dtype=complex)

    if(fast):
        print('using faster fast algorithm lol')
        s_mat, h_mat = rtl_helpers.get_sr_mats_fast(initial_ref, inital_dt,
                                                    Ninitial_refs, mol.get_hamiltonian(),
                                                    nqubits)

        # print('using slower fast algorithm lol')
        # for p in range(Ninitial_refs):
        #     for q in range(p, Ninitial_refs):
        #         h_mat[p][q] = rtl_helpers.matrix_element_fast(initial_ref, inital_dt, p, q, mol.get_hamiltonian(),
        #                                         nqubits, mol.get_hamiltonian())
        #         h_mat[q][p] = np.conj(h_mat[p][q])
        #
        #         s_mat[p][q] = rtl_helpers.matrix_element_fast(initial_ref, inital_dt, p, q, mol.get_hamiltonian(),
        #                                         nqubits)
        #         s_mat[q][p] = np.conj(s_mat[p][q])


    else:
        for p in range(Ninitial_refs):
            for q in range(p, Ninitial_refs):
                h_mat[p][q] = rtl_helpers.matrix_element(initial_ref, inital_dt, p, q, mol.get_hamiltonian(),
                                                nqubits, mol.get_hamiltonian())
                h_mat[q][p] = np.conj(h_mat[p][q])

                s_mat[p][q] = rtl_helpers.matrix_element(initial_ref, inital_dt, p, q, mol.get_hamiltonian(),
                                                nqubits)
                s_mat[q][p] = np.conj(s_mat[p][q])

    print("\nS initial:\n")
    rtl_helpers.matprint(s_mat)

    print("\nHbar: initial\n")
    rtl_helpers.matprint(h_mat)

    if(np.linalg.cond(s_mat) > 1.0e7):
        raise ValueError('cond(S) > 1e7, matrix is possibly ill conditioned, use larger inital_dt.')

    evals, evecs = linalg.eig(h_mat,s_mat)

    # print('\nARTLanczos (unsorted!) initial evals:\n\n', evals)
    # print('\nARTLanczos initial evecs:\n')
    # matprint(evecs)

    # need to make sorted evals and evecs...
    # use sorted_largest_idxs()
    sorted_evals_idxs = sorted_largest_idxs(evals, use_real=True, rev=False)
    sorted_evals = np.zeros((Ninitial_refs), dtype=complex)
    sorted_evecs = np.zeros((Ninitial_refs,Ninitial_refs), dtype=complex)
    for n in range(Ninitial_refs):
        old_idx = sorted_evals_idxs[n][1]
        sorted_evals[n]   = evals[old_idx]
        sorted_evecs[:,n] = evecs[:,old_idx]

    print('\nARTLanczos (sorted!) initial evals:\n\n', sorted_evals)
    print('\nARTLanczos initial sorted evecs:\n')
    matprint(sorted_evecs)


    sq_mod_evecs = np.zeros((Ninitial_refs,Ninitial_refs), dtype=complex)

    for p in range(Ninitial_refs):
        for q in range(Ninitial_refs):
            sq_mod_evecs[p][q] = sorted_evecs[p][q] * np.conj(sorted_evecs[p][q])

    print('\nARTLanczos initial evecs square modulous:\n')
    matprint(sq_mod_evecs)

    basis_coeff_vec_lst = []
    for n in range(Ninitial_refs):
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
    Cprime = (sorted_evecs.transpose()).dot(basis_coeff_mat)
    for n in range(Ninitial_refs):
        for i, val in enumerate(Cprime[n]):
            Cprime[n][i] *= np.conj(val)


    # Now using the approximation
    for n in range(Ninitial_refs):
        for i, val in enumerate(basis_coeff_vec_lst[n]):
            basis_coeff_vec_lst[n][i] *= np.conj(val)


    Cprime_sq_mod = (sq_mod_evecs.transpose()).dot(basis_coeff_mat)

    true_idx_lst = []
    idx_lst = []
    if(target_root is not None):
        print('\nTargeting refs for root ', target_root)
        true_sorted_idxs = sorted_largest_idxs(Cprime[target_root,:])
        sorted_idxs = sorted_largest_idxs(Cprime_sq_mod[target_root,:])
        for n in range(Ninitial_refs):
            idx_lst.append( sorted_idxs[n][1] )
            true_idx_lst.append( true_sorted_idxs[n][1] )

    else:
        for n in range(Ninitial_refs):
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

    for idx in idx_lst:
        initial_ref_lst.append(intiger_to_ref(idx, nqubits))

    return initial_ref_lst
