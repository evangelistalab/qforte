import qforte
from qforte.utils import trotterization as trot

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

def ref_to_basis_idx(ref):
    temp = ref.copy()
    temp.reverse()
    return int("".join(str(x) for x in temp), 2)

def matrix_element(ref, dt, m, n, H, nqubits, A = None, trot_order=1):
    """
    This function returns a single matrix element M_bk based on the evolutio of
    two unitary operators Ub = exp(-i * m * dt * H) and H_q = exp(-i * n * dt *H) on a
    reference state |Phi_o>.

    :param ref: a list representing the referende state |Phi_o>
    :param dt: a double representing the real time step
    :param m: the intager number of time steps for the Ub evolution
    :param n: the intager number of time steps for the Uk evolution
    :param H: the QuantumOperator to time evolove under
    :param nqubits: the intager number of qubits
    :param A: (optional) the overal operator to measure with respect to
    """
    value = 0.0
    ancilla_idx = nqubits

    Uk = qforte.QuantumCircuit()

    temp_op1 = qforte.QuantumOperator()
    for t in H.terms():
        c, op = t
        phase = -1.0j * n * dt * c
        temp_op1.add_term(phase, op)

    expn_op1, phase1 = qforte.trotterization.trotterize_w_cRz(temp_op1, ancilla_idx, trotter_order=trot_order)

    for gate in expn_op1.gates():
        Uk.add_gate(gate)

    Ub = qforte.QuantumCircuit()

    temp_op2 = qforte.QuantumOperator()
    for t in H.terms():
        c, op = t
        phase = -1.0j * m * dt * c
        temp_op2.add_term(phase, op)

    expn_op2, phase2 = qforte.trotterization.trotterize_w_cRz(temp_op2, ancilla_idx, trotter_order=trot_order, Use_open_cRz=False)

    for gate in expn_op2.gates():
        Ub.add_gate(gate)

    if A == None:
        cir = qforte.QuantumCircuit()
        for j in range(nqubits):
            if ref[j] == 1:
                cir.add_gate(qforte.make_gate('X', j, j))

        cir.add_gate(qforte.make_gate('H', ancilla_idx, ancilla_idx))

        cir.add_circuit(Uk)

        cir.add_gate(qforte.make_gate('X', ancilla_idx, ancilla_idx))
        cir.add_circuit(Ub)
        cir.add_gate(qforte.make_gate('X', ancilla_idx, ancilla_idx))

        X_op = qforte.QuantumOperator()
        x_circ = qforte.QuantumCircuit()
        Y_op = qforte.QuantumOperator()
        y_circ = qforte.QuantumCircuit()

        x_circ.add_gate(qforte.make_gate('X', ancilla_idx, ancilla_idx))
        y_circ.add_gate(qforte.make_gate('Y', ancilla_idx, ancilla_idx))

        X_op.add_term(1.0, x_circ)
        Y_op.add_term(1.0, y_circ)

        X_exp = qforte.Experiment(nqubits+1, cir, X_op, 100)
        Y_exp = qforte.Experiment(nqubits+1, cir, Y_op, 100)

        params = [1.0]
        x_value = X_exp.perfect_experimental_avg(params)
        y_value = Y_exp.perfect_experimental_avg(params)

        value = (x_value + 1.0j * y_value) * phase1 * np.conj(phase2)


    else:
        value = 0.0
        for t in A.terms():
            c, V_l = t

            # TODO: Optemize (Nick)
            cV_l = qforte.QuantumCircuit()
            for gate in V_l.gates():
                gate_str = gate.gate_id()
                target = gate.target()
                control_gate_str = 'c' + gate_str
                cV_l.add_gate(qforte.make_gate(control_gate_str, target, ancilla_idx))

            cir = qforte.QuantumCircuit()
            for j in range(nqubits):
                if ref[j] == 1:
                    cir.add_gate(qforte.make_gate('X', j, j))

            cir.add_gate(qforte.make_gate('H', ancilla_idx, ancilla_idx))

            cir.add_circuit(Uk)
            cir.add_circuit(cV_l)

            cir.add_gate(qforte.make_gate('X', ancilla_idx, ancilla_idx))
            cir.add_circuit(Ub)
            cir.add_gate(qforte.make_gate('X', ancilla_idx, ancilla_idx))

            X_op = qforte.QuantumOperator()
            x_circ = qforte.QuantumCircuit()
            Y_op = qforte.QuantumOperator()
            y_circ = qforte.QuantumCircuit()

            x_circ.add_gate(qforte.make_gate('X', ancilla_idx, ancilla_idx))
            y_circ.add_gate(qforte.make_gate('Y', ancilla_idx, ancilla_idx))

            X_op.add_term(1.0, x_circ)
            Y_op.add_term(1.0, y_circ)

            X_exp = qforte.Experiment(nqubits+1, cir, X_op, 100)
            Y_exp = qforte.Experiment(nqubits+1, cir, Y_op, 100)

            # TODO: Remove params are required arg (Nick)
            params = [1.0]
            x_value = X_exp.perfect_experimental_avg(params)
            y_value = Y_exp.perfect_experimental_avg(params)

            element = (x_value + 1.0j * y_value) * phase1 * np.conj(phase2)
            value += c * element

    return value

def get_sr_mats_fast(ref, dt, nstates, H, nqubits, trot_order=1):
    """
    This function returns a single matrix element M_bk based on the evolutio of
    two unitary operators Ub = exp(-i * m * dt * H) and H_q = exp(-i * n * dt *H) on a
    reference state |Phi_o>. This is done WITHOUT measuring any operators,
    but rather computes the expecation value directly using a priori knowlege of
    the wavefunction coefficients

    :param ref: a list representing the referende state |Phi_o>
    :param dt: a double representing the real time step
    :param H: the QuantumOperator to time evolove under
    :param nqubits: the intager number of qubits
    :param A: (optional) the overal operator to measure with respect to
    """

    h_mat = np.zeros((nstates,nstates), dtype=complex)
    s_mat = np.zeros((nstates,nstates), dtype=complex)

    omega_lst = []
    Homega_lst = []
    for n in range(nstates):

        Un = qforte.QuantumCircuit()
        for j in range(nqubits):
            if ref[j] == 1:
                Un.add_gate(qforte.make_gate('X', j, j))
                phase1 = 1.0

        if(n>0):
            temp_op1 = qforte.QuantumOperator()
            for t in H.terms():
                c, op = t
                phase = -1.0j * n * dt * c
                temp_op1.add_term(phase, op)

            expn_op1, phase1 = qforte.trotterization.trotterize(temp_op1, trotter_order=trot_order)
            Un.add_circuit(expn_op1)

        QC = qforte.QuantumComputer(nqubits)
        QC.apply_circuit(Un)
        QC.apply_constant(phase1)
        omega_lst.append(np.asarray(QC.get_coeff_vec(), dtype=complex))

        Homega = np.zeros((2**nqubits), dtype=complex)

        for k in range(len(H.terms())):
            QCk = qforte.QuantumComputer(nqubits)
            QCk.set_coeff_vec(QC.get_coeff_vec())

            if(H.terms()[k][1] is not None):
                QCk.apply_circuit(H.terms()[k][1])
            if(H.terms()[k][0] is not None):
                QCk.apply_constant(H.terms()[k][0])

            Homega = np.add(Homega, np.asarray(QCk.get_coeff_vec(), dtype=complex))

        Homega_lst.append(Homega)

    for p in range(nstates):
        for q in range(p, nstates):
            h_mat[p][q] = np.vdot(omega_lst[p], Homega_lst[q])
            h_mat[q][p] = np.conj(h_mat[p][q])
            s_mat[p][q] = np.vdot(omega_lst[p], omega_lst[q])
            s_mat[q][p] = np.conj(s_mat[p][q])

    return s_mat, h_mat


def mr_matrix_element(ref_I, ref_J, dt_I, dt_J, m, n, H, nqubits, A = None, trot_order=1):
    """
    This function returns a single matrix element M_bk based on the evolutio of
    two unitary operators Ub = exp(-i * m * dt * H) and H_q = exp(-i * n * dt *H) on a
    reference state |Phi_o>.

    :param ref: a list representing the referende state |Phi_o>
    :param dt: a double representing the real time step
    :param m: the intager number of time steps for the Ub evolution
    :param n: the intager number of time steps for the Uk evolution
    :param H: the QuantumOperator to time evolove under
    :param nqubits: the intager number of qubits
    :param A: (optional) the overal operator to measure with respect to
    """
    value = 0.0
    ancilla_idx = nqubits

    Uk = qforte.QuantumCircuit()
    for i in range(nqubits):
        if ref_I[i] == 1:
            Uk.add_gate(qforte.make_gate('cX', i, ancilla_idx))

    temp_op1 = qforte.QuantumOperator()
    for t in H.terms():
        c, op = t
        phase = -1.0j * (m) * dt_I * c
        temp_op1.add_term(phase, op)

    expn_op1, phase1 = qforte.trotterization.trotterize_w_cRz(temp_op1, ancilla_idx, trotter_order=trot_order)
    for gate in expn_op1.gates():
        Uk.add_gate(gate)

    Ub = qforte.QuantumCircuit()
    for j in range(nqubits):
        if ref_J[j] == 1:
            Ub.add_gate(qforte.make_gate('cX', j, ancilla_idx))

    temp_op2 = qforte.QuantumOperator()
    for t in H.terms():
        c, op = t
        phase = -1.0j * (n) * dt_J * c
        temp_op2.add_term(phase, op)

    expn_op2, phase2 = qforte.trotterization.trotterize_w_cRz(temp_op2, ancilla_idx, trotter_order=trot_order, Use_open_cRz=False)
    for gate in expn_op2.gates():
        Ub.add_gate(gate)

    if A == None:
        cir = qforte.QuantumCircuit()
        cir.add_gate(qforte.make_gate('H', ancilla_idx, ancilla_idx))
        cir.add_circuit(Uk)
        cir.add_gate(qforte.make_gate('X', ancilla_idx, ancilla_idx))
        cir.add_circuit(Ub)
        cir.add_gate(qforte.make_gate('X', ancilla_idx, ancilla_idx))

        X_op = qforte.QuantumOperator()
        x_circ = qforte.QuantumCircuit()
        Y_op = qforte.QuantumOperator()
        y_circ = qforte.QuantumCircuit()

        x_circ.add_gate(qforte.make_gate('X', ancilla_idx, ancilla_idx))
        y_circ.add_gate(qforte.make_gate('Y', ancilla_idx, ancilla_idx))

        X_op.add_term(1.0, x_circ)
        Y_op.add_term(1.0, y_circ)

        X_exp = qforte.Experiment(nqubits+1, cir, X_op, 100)
        Y_exp = qforte.Experiment(nqubits+1, cir, Y_op, 100)

        params = [1.0]
        x_value = X_exp.perfect_experimental_avg(params)
        y_value = Y_exp.perfect_experimental_avg(params)

        value = (x_value + 1.0j * y_value) * phase1 * np.conj(phase2)

    else:
        value = 0.0
        for t in A.terms():
            c, V_l = t

            # TODO: Optemize (Nick)
            cV_l = qforte.QuantumCircuit()
            for gate in V_l.gates():
                gate_str = gate.gate_id()
                target = gate.target()
                control_gate_str = 'c' + gate_str
                cV_l.add_gate(qforte.make_gate(control_gate_str, target, ancilla_idx))

            cir = qforte.QuantumCircuit()
            cir.add_gate(qforte.make_gate('H', ancilla_idx, ancilla_idx))

            cir.add_circuit(Uk)
            cir.add_circuit(cV_l)

            cir.add_gate(qforte.make_gate('X', ancilla_idx, ancilla_idx))
            cir.add_circuit(Ub)
            cir.add_gate(qforte.make_gate('X', ancilla_idx, ancilla_idx))

            X_op = qforte.QuantumOperator()
            x_circ = qforte.QuantumCircuit()
            Y_op = qforte.QuantumOperator()
            y_circ = qforte.QuantumCircuit()

            x_circ.add_gate(qforte.make_gate('X', ancilla_idx, ancilla_idx))
            y_circ.add_gate(qforte.make_gate('Y', ancilla_idx, ancilla_idx))

            X_op.add_term(1.0, x_circ)
            Y_op.add_term(1.0, y_circ)

            X_exp = qforte.Experiment(nqubits+1, cir, X_op, 100)
            Y_exp = qforte.Experiment(nqubits+1, cir, Y_op, 100)

            params = [1.0]
            x_value = X_exp.perfect_experimental_avg(params)
            y_value = Y_exp.perfect_experimental_avg(params)

            element = (x_value + 1.0j * y_value) * phase1 * np.conj(phase2)
            value += c * element

    return value

def get_mr_mats_fast(ref_lst, nstates_per_ref, dt_lst, H, nqubits, trot_order=1):
    """
    This function returns a single matrix element M_bk based on the evolutio of
    two unitary operators Ub = exp(-i * m * dt * H) and H_q = exp(-i * n * dt *H) on a
    reference state |Phi_o>. This is done WITHOUT measuring any operators,
    but rather computes the expecation value directly using a priori knowlege of
    the wavefunction coefficients

    :param ref: a list representing the referende state |Phi_o>
    :param dt: a double representing the real time step
    :param H: the QuantumOperator to time evolove under
    :param nqubits: the intager number of qubits
    :param A: (optional) the overal operator to measure with respect to
    """

    num_tot_basis = len(ref_lst) * nstates_per_ref

    h_mat = np.zeros((num_tot_basis,num_tot_basis), dtype=complex)
    s_mat = np.zeros((num_tot_basis,num_tot_basis), dtype=complex)

    omega_lst = []
    Homega_lst = []

    for i, ref in enumerate(ref_lst):
        dt = dt_lst[i]
        for n in range(nstates_per_ref):

            Un = qforte.QuantumCircuit()
            for j in range(nqubits):
                if ref[j] == 1:
                    Un.add_gate(qforte.make_gate('X', j, j))
                    phase1 = 1.0

            if(n>0):
                temp_op1 = qforte.QuantumOperator()
                for t in H.terms():
                    c, op = t
                    phase = -1.0j * n * dt * c
                    temp_op1.add_term(phase, op)

                expn_op1, phase1 = qforte.trotterization.trotterize(temp_op1, trotter_order=trot_order)
                Un.add_circuit(expn_op1)

            QC = qforte.QuantumComputer(nqubits)
            QC.apply_circuit(Un)
            QC.apply_constant(phase1)
            omega_lst.append(np.asarray(QC.get_coeff_vec(), dtype=complex))

            Homega = np.zeros((2**nqubits), dtype=complex)

            for k in range(len(H.terms())):
                QCk = qforte.QuantumComputer(nqubits)
                QCk.set_coeff_vec(QC.get_coeff_vec())

                if(H.terms()[k][1] is not None):
                    QCk.apply_circuit(H.terms()[k][1])
                if(H.terms()[k][0] is not None):
                    QCk.apply_constant(H.terms()[k][0])

                Homega = np.add(Homega, np.asarray(QCk.get_coeff_vec(), dtype=complex))

            Homega_lst.append(Homega)

    for p in range(num_tot_basis):
        for q in range(p, num_tot_basis):
            h_mat[p][q] = np.vdot(omega_lst[p], Homega_lst[q])
            h_mat[q][p] = np.conj(h_mat[p][q])
            s_mat[p][q] = np.vdot(omega_lst[p], omega_lst[q])
            s_mat[q][p] = np.conj(s_mat[p][q])

    return s_mat, h_mat

def get_sa_mr_mats_fast(ref_lst, nstates_per_ref, dt_lst, H, nqubits, trot_order=1):
    """
    This function returns a single matrix element

    :param ref: a list representing the referende state |Phi_o>
    :param dt: a double representing the real time step
    :param H: the QuantumOperator to time evolove under
    :param nqubits: the intager number of qubits
    :param A: (optional) the overal operator to measure with respect to
    """

    num_tot_basis = len(ref_lst) * nstates_per_ref

    h_mat = np.zeros((num_tot_basis,num_tot_basis), dtype=complex)
    s_mat = np.zeros((num_tot_basis,num_tot_basis), dtype=complex)

    omega_lst = []
    Homega_lst = []

    for i, ref in enumerate(ref_lst):
        dt = dt_lst[i]
        for n in range(nstates_per_ref):

            Un = qforte.QuantumCircuit()

            phase1 = 1.0
            if(n>0):
                temp_op1 = qforte.QuantumOperator()
                for t in H.terms():
                    c, op = t
                    phase = -1.0j * n * dt * c
                    temp_op1.add_term(phase, op)

                expn_op1, phase1 = qforte.trotterization.trotterize(temp_op1, trotter_order=trot_order)
                Un.add_circuit(expn_op1)

            QC = qforte.QuantumComputer(nqubits)

            state_prep_lst = []
            for term in ref:
                coeff = term[0]
                det = term[1]
                idx = ref_to_basis_idx(det)
                state = qforte.QuantumBasis(idx)
                state_prep_lst.append( (state, coeff) )

            QC.set_state(state_prep_lst)
            QC.apply_circuit(Un)
            QC.apply_constant(phase1)
            omega_lst.append(np.asarray(QC.get_coeff_vec(), dtype=complex))

            Homega = np.zeros((2**nqubits), dtype=complex)

            for k in range(len(H.terms())):

                QCk = qforte.QuantumComputer(nqubits)
                QCk.set_coeff_vec(QC.get_coeff_vec())
                if(H.terms()[k][1] is not None):
                    QCk.apply_circuit(H.terms()[k][1])

                if(H.terms()[k][0] is not None):
                    QCk.apply_constant(H.terms()[k][0])

                Homega = np.add(Homega, np.asarray(QCk.get_coeff_vec(), dtype=complex))

            Homega_lst.append(Homega)

    for p in range(num_tot_basis):
        for q in range(p, num_tot_basis):
            h_mat[p][q] = np.vdot(omega_lst[p], Homega_lst[q])
            h_mat[q][p] = np.conj(h_mat[p][q])
            s_mat[p][q] = np.vdot(omega_lst[p], omega_lst[q])
            s_mat[q][p] = np.conj(s_mat[p][q])

    return s_mat, h_mat

def canonical_geig_solve(S, H, print_mats=False, sort_ret_vals=False):

    THRESHOLD = 1e-7
    s, U = linalg.eig(S)
    s_prime = []

    for sii in s:
        if(np.imag(sii) > 1e-12):
            raise ValueError('S may not be hermetian, large imag. eval component.')
        if(np.real(sii) > THRESHOLD):
            s_prime.append(np.real(sii))

    if((len(s) - len(s_prime)) != 0):
        print('\nGeneralized eigenvalue probelm rank was reduced, matrix may be ill conditioned!')
        print('  s is of inital rank:    ', len(s))
        print('  s is of truncated rank: ', len(s_prime))

    X_prime = np.zeros((len(s), len(s_prime)), dtype=complex)
    for i in range(len(s)):
        for j in range(len(s_prime)):
            X_prime[i][j] = U[i][j] / np.sqrt(s_prime[j])

    H_prime = (((X_prime.conjugate()).transpose()).dot(H)).dot(X_prime)
    e_prime, C_prime = linalg.eig(H_prime)
    C = X_prime.dot(C_prime)

    if(print_mats):
        print('\n      -----------------------------')
        print('      Printing GEVS Mats (unsorted)')
        print('      -----------------------------')

        I_prime = (((C.conjugate()).transpose()).dot(S)).dot(C)

        print('\ns:\n')
        print(s)
        print('\nU:\n')
        matprint(U)
        print('\nX_prime:\n')
        matprint(X_prime)
        print('\nH_prime:\n')
        matprint(H_prime)
        print('\ne_prime:\n')
        print(e_prime)
        print('\nC_prime:\n')
        matprint(C_prime)
        print('\ne_prime:\n')
        print(e_prime)
        print('\nC:\n')
        matprint(C)
        print('\nIprime:\n')
        matprint(I_prime)

        print('\n      ------------------------------')
        print('          Printing GEVS Mats End    ')
        print('      ------------------------------')

    if(sort_ret_vals):
        sorted_e_prime_idxs = sorted_largest_idxs(e_prime, use_real=True, rev=False)
        sorted_e_prime = np.zeros((len(e_prime)), dtype=complex)
        sorted_C_prime = np.zeros((len(e_prime),len(e_prime)), dtype=complex)
        sorted_X_prime = np.zeros((len(s),len(e_prime)), dtype=complex)
        for n in range(len(e_prime)):
            old_idx = sorted_e_prime_idxs[n][1]
            sorted_e_prime[n]   = e_prime[old_idx]
            sorted_C_prime[:,n] = C_prime[:,old_idx]
            sorted_X_prime[:,n] = X_prime[:,old_idx]

        sorted_C = sorted_X_prime.dot(sorted_C_prime)
        return sorted_e_prime, sorted_C

    else:
        return e_prime, C
