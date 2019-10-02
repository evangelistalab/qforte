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


def sorted_largest_idxs(array, rev=True):
        temp = np.empty((len(array)), dtype=object )
        for i, val in enumerate(array):
            temp[i] = (val, i)
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
        for p in range(Ninitial_refs):
            for q in range(p, Ninitial_refs):
                h_mat[p][q] = rtl_helpers.matrix_element_fast(initial_ref, inital_dt, p, q, mol.get_hamiltonian(),
                                                nqubits, mol.get_hamiltonian())
                h_mat[q][p] = np.conj(h_mat[p][q])

                s_mat[p][q] = rtl_helpers.matrix_element_fast(initial_ref, inital_dt, p, q, mol.get_hamiltonian(),
                                                nqubits)
                s_mat[q][p] = np.conj(s_mat[p][q])


    else:
        for p in range(Ninitial_refs):
            for q in range(p, Ninitial_refs):
                h_mat[p][q] = rtl_helpers.matrix_element(initial_ref, inital_dt, p, q, mol.get_hamiltonian(),
                                                nqubits, mol.get_hamiltonian())
                h_mat[q][p] = np.conj(h_mat[p][q])

                s_mat[p][q] = rtl_helpers.matrix_element(initial_ref, inital_dt, p, q, mol.get_hamiltonian(),
                                                nqubits)
                s_mat[q][p] = np.conj(s_mat[p][q])



    if(np.linalg.cond(s_mat) > 1.0e7):
        raise ValueError('cond(S) > 1e7, matrix is possibly ill conditioned, use larger inital_dt.')

    evals, evecs = linalg.eig(h_mat,s_mat)

    print('\nARTLanczos (unsorted!) initial evals:\n\n', evals)
    print('\nARTLanczos initial evecs:\n')
    matprint(evecs)

    sq_mod_evecs = np.zeros((Ninitial_refs,Ninitial_refs), dtype=complex)

    for p in range(Ninitial_refs):
        for q in range(Ninitial_refs):
            sq_mod_evecs[p][q] = evecs[p][q] * np.conj(evecs[p][q])

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

    for n in range(Ninitial_refs):
        for i, val in enumerate(basis_coeff_vec_lst[n]):
            basis_coeff_vec_lst[n][i] *= np.conj(val)

    basis_coeff_mat = np.array(basis_coeff_vec_lst)
    Cprime = (sq_mod_evecs).dot(basis_coeff_mat)

    idx_lst = []

    if(target_root is not None):
        print('\nTargeting refs for root ', target_root)
        sorted_idxs = sorted_largest_idxs(Cprime[target_root,:])
        for n in range(Ninitial_refs):
            idx_lst.append( sorted_idxs[n][1] )

    else:
        for n in range(Ninitial_refs):
            sorted_idxs = sorted_largest_idxs(Cprime[n,:])
            if(sorted_idxs[0][1] in idx_lst):
                k = 1
                while(sorted_idxs[k][1] in idx_lst):
                    k += 1
                    if(k == 2**nqubits):
                        raise ValueError('Selection of inital determinant references unsucessful.')

                idx_lst.append( sorted_idxs[k][1] )

            else:
                idx_lst.append( sorted_idxs[0][1] )

    print('\nInitial ref guesses:\n')
    print('root               dominant determinant  ')
    print('----------------------------------------')
    for i, idx in enumerate(idx_lst):
        basis = qforte.QuantumBasis(idx)
        print('  ', i, '                ', basis.str(nqubits))

    for idx in idx_lst:
        initial_ref_lst.append(intiger_to_ref(idx, nqubits))

    return initial_ref_lst
