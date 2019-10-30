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

    # Prepare Uk, the right controlled unitary circuit exp(-i n dt H)
    Uk = qforte.QuantumCircuit()

    # Make prod_l exp(-i n dt h_l P_l)
    temp_op1 = qforte.QuantumOperator() # A temporary operator to multiply H by
    for t in H.terms():
        c, op = t
        phase = -1.0j * n * dt * c
        temp_op1.add_term(phase, op)

    # Trotterize with controlled-Rz(2*theta) gates (as per the last circuit of
    # section 3.3 in Mario's notes)
    expn_op1, phase1 = qforte.trotterization.trotterize_w_cRz(temp_op1, ancilla_idx, trotter_order=trot_order)

    for gate in expn_op1.gates():
        Uk.add_gate(gate)

    # Prepare Ub^daggar the left open-controlled unitary circuit exp(+i n dt H) prod_j X_j
    Ub = qforte.QuantumCircuit()
    # Make rev_prod_k exp(i n dt h_k P_k)
    temp_op2 = qforte.QuantumOperator()
    for t in H.terms():
        c, op = t
        phase = -1.0j * m * dt * c
        temp_op2.add_term(phase, op)

    # Trotterize with open-controlled-Rz(2*theta) gates (as per the last circuit of
    # section 3.3 in Mario's notes). Open-controlled gates (denoted as a controll
    # qubit with a white rather than black dot) are performed by ading an X gate before
    # and after the controll gate (See Fig. 11 on page 185 of Nielson and Chung)
    expn_op2, phase2 = qforte.trotterization.trotterize_w_cRz(temp_op2, ancilla_idx, trotter_order=trot_order, Use_open_cRz=False)

    for gate in expn_op2.gates():
        Ub.add_gate(gate)

    if A == None:
        # 1. Initialize State to |Psi_o>
        cir = qforte.QuantumCircuit()
        for j in range(nqubits):
            if ref[j] == 1:
                cir.add_gate(qforte.make_gate('X', j, j))

        # For details on steps 2-4, refer to Mario's notes (section 3.3) and
        # Ref. [3] therein: https://arxiv.org/pdf/quant-ph/0304063.pdf

        # 2. Split Ancilla qubit
        cir.add_gate(qforte.make_gate('H', ancilla_idx, ancilla_idx))

        # 3. Apply controlled unitaries ()
        cir.add_circuit(Uk)

        cir.add_gate(qforte.make_gate('X', ancilla_idx, ancilla_idx))
        cir.add_circuit(Ub)
        cir.add_gate(qforte.make_gate('X', ancilla_idx, ancilla_idx))


        # 4. Measure X and Y oporators on ancilla qubit
        X_op = qforte.QuantumOperator()
        x_circ = qforte.QuantumCircuit()
        Y_op = qforte.QuantumOperator()
        y_circ = qforte.QuantumCircuit()

        x_circ.add_gate(qforte.make_gate('X', ancilla_idx, ancilla_idx))
        y_circ.add_gate(qforte.make_gate('Y', ancilla_idx, ancilla_idx))

        X_op.add_term(1.0, x_circ)
        Y_op.add_term(1.0, y_circ)

        # Initalize experiment
        X_exp = qforte.Experiment(nqubits+1, cir, X_op, 100)
        Y_exp = qforte.Experiment(nqubits+1, cir, Y_op, 100)

        params = [1.0]
        x_value = X_exp.perfect_experimental_avg(params)
        y_value = Y_exp.perfect_experimental_avg(params)

        # <Psi_o|Ub^dag Uk|Psi_o> = <2* sigma_+> = <Psi_f|X|Psi_f> + * i * <Psi_f|Y|Psi_f>
        value = (x_value + 1.0j * y_value) * phase1 * np.conj(phase2)


    else:
        value = 0.0
        for t in A.terms():
            c, V_l = t

            # Make controlled cV_l
            # TODO: Optemize (Nick)
            cV_l = qforte.QuantumCircuit()
            for gate in V_l.gates():
                gate_str = gate.gate_id()
                target = gate.target()
                control_gate_str = 'c' + gate_str
                cV_l.add_gate(qforte.make_gate(control_gate_str, target, ancilla_idx))


            # 1. Initialize State to |Psi_o>
            cir = qforte.QuantumCircuit()
            for j in range(nqubits):
                if ref[j] == 1:
                    cir.add_gate(qforte.make_gate('X', j, j))

            # For details on steps 2-4, refer to Mario's notes (section 3.3) and
            # Ref. [3] therein: https://arxiv.org/pdf/quant-ph/0304063.pdf

            # 2. Split Ancilla qubit
            cir.add_gate(qforte.make_gate('H', ancilla_idx, ancilla_idx))

            # 3. Add controlled unitaries and the lth term in the hamiltionan V_l
            cir.add_circuit(Uk)
            cir.add_circuit(cV_l)

            cir.add_gate(qforte.make_gate('X', ancilla_idx, ancilla_idx))
            cir.add_circuit(Ub)
            cir.add_gate(qforte.make_gate('X', ancilla_idx, ancilla_idx))


            # 4. Measure X and Y oporators on ancilla qubit
            X_op = qforte.QuantumOperator()
            x_circ = qforte.QuantumCircuit()
            Y_op = qforte.QuantumOperator()
            y_circ = qforte.QuantumCircuit()

            x_circ.add_gate(qforte.make_gate('X', ancilla_idx, ancilla_idx))
            y_circ.add_gate(qforte.make_gate('Y', ancilla_idx, ancilla_idx))

            X_op.add_term(1.0, x_circ)
            Y_op.add_term(1.0, y_circ)

            # Initalize experiment
            X_exp = qforte.Experiment(nqubits+1, cir, X_op, 100)
            Y_exp = qforte.Experiment(nqubits+1, cir, Y_op, 100)

            params = [1.0]
            x_value = X_exp.perfect_experimental_avg(params)
            y_value = Y_exp.perfect_experimental_avg(params)

            # <Psi_o|Ub V_l Uk|Psi_o> = <2* sigma_+> = <Psi_f|X|Psi_f> + * i * <Psi_f|Y|Psi_f>
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

    ############################################################################

    #Initialize arrays for Hbar and S
    h_mat = np.zeros((nstates,nstates), dtype=complex)
    s_mat = np.zeros((nstates,nstates), dtype=complex)

    #Build |Ωn> and |H Ωn> Vectors
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

    # Prepare Uk, the right controlled unitary circuit exp(-i n dt H)
    Uk = qforte.QuantumCircuit()
    for i in range(nqubits):
        if ref_I[i] == 1:
            Uk.add_gate(qforte.make_gate('cX', i, ancilla_idx))

    # Make prod_l exp(-i n dt h_l P_l)
    temp_op1 = qforte.QuantumOperator() # A temporary operator to multiply H by
    for t in H.terms():
        c, op = t
        phase = -1.0j * (m) * dt_I * c
        temp_op1.add_term(phase, op)

    # Trotterize with controlled-Rz(2*theta) gates (as per the last circuit of
    # section 3.3 in Mario's notes)
    expn_op1, phase1 = qforte.trotterization.trotterize_w_cRz(temp_op1, ancilla_idx, trotter_order=trot_order)
    for gate in expn_op1.gates():
        Uk.add_gate(gate)

    # Prepare Ub^daggar the left open-controlled unitary circuit exp(+i n dt H) prod_j X_j
    Ub = qforte.QuantumCircuit()
    for j in range(nqubits):
        if ref_J[j] == 1:
            Ub.add_gate(qforte.make_gate('cX', j, ancilla_idx))

    # Make rev_prod_k exp(i n dt h_k P_k)
    temp_op2 = qforte.QuantumOperator()
    for t in H.terms():
        c, op = t
        phase = -1.0j * (n) * dt_J * c
        temp_op2.add_term(phase, op)

    # Trotterize with open-controlled-Rz(2*theta). Open-controlled gates (denoted as a controll
    # qubit with a white rather than black dot) are performed by ading an X gate before
    # and after the controll gate (See Fig. 11 on page 185 of Nielson and Chung)
    expn_op2, phase2 = qforte.trotterization.trotterize_w_cRz(temp_op2, ancilla_idx, trotter_order=trot_order, Use_open_cRz=False)
    for gate in expn_op2.gates():
        Ub.add_gate(gate)

    if A == None:
        # 1. Initialize State to |Psi_o>
        cir = qforte.QuantumCircuit()

        # For details on steps 2-4, refer to Mario's notes (section 3.3) and
        # Ref. [3] therein: https://arxiv.org/pdf/quant-ph/0304063.pdf

        # 2. Split Ancilla qubit
        cir.add_gate(qforte.make_gate('H', ancilla_idx, ancilla_idx))

        # 3. Apply controlled unitaries ()
        cir.add_circuit(Uk)

        cir.add_gate(qforte.make_gate('X', ancilla_idx, ancilla_idx))
        cir.add_circuit(Ub)
        cir.add_gate(qforte.make_gate('X', ancilla_idx, ancilla_idx))


        # 4. Measure X and Y oporators on ancilla qubit
        X_op = qforte.QuantumOperator()
        x_circ = qforte.QuantumCircuit()
        Y_op = qforte.QuantumOperator()
        y_circ = qforte.QuantumCircuit()

        x_circ.add_gate(qforte.make_gate('X', ancilla_idx, ancilla_idx))
        y_circ.add_gate(qforte.make_gate('Y', ancilla_idx, ancilla_idx))

        X_op.add_term(1.0, x_circ)
        Y_op.add_term(1.0, y_circ)

        # Initalize experiment
        X_exp = qforte.Experiment(nqubits+1, cir, X_op, 100)
        Y_exp = qforte.Experiment(nqubits+1, cir, Y_op, 100)

        params = [1.0]
        x_value = X_exp.perfect_experimental_avg(params)
        y_value = Y_exp.perfect_experimental_avg(params)

        # <Psi_o|Ub^dag Uk|Psi_o> = <2* sigma_+> = <Psi_f|X|Psi_f> + * i * <Psi_f|Y|Psi_f>
        value = (x_value + 1.0j * y_value) * phase1 * np.conj(phase2)


    else:
        value = 0.0
        for t in A.terms():
            c, V_l = t

            # Make controlled cV_l
            # TODO: Optemize (Nick)
            cV_l = qforte.QuantumCircuit()
            for gate in V_l.gates():
                gate_str = gate.gate_id()
                target = gate.target()
                control_gate_str = 'c' + gate_str
                cV_l.add_gate(qforte.make_gate(control_gate_str, target, ancilla_idx))


            # 1. Initialize State to |Psi_o>
            cir = qforte.QuantumCircuit()

            # For details on steps 2-4, refer to Mario's notes (section 3.3) and
            # Ref. [3] therein: https://arxiv.org/pdf/quant-ph/0304063.pdf

            # 2. Split Ancilla qubit
            cir.add_gate(qforte.make_gate('H', ancilla_idx, ancilla_idx))

            # 3. Add controlled unitaries and the lth term in the hamiltionan V_l
            cir.add_circuit(Uk)
            cir.add_circuit(cV_l)

            cir.add_gate(qforte.make_gate('X', ancilla_idx, ancilla_idx))
            cir.add_circuit(Ub)
            cir.add_gate(qforte.make_gate('X', ancilla_idx, ancilla_idx))


            # 4. Measure X and Y oporators on ancilla qubit
            X_op = qforte.QuantumOperator()
            x_circ = qforte.QuantumCircuit()
            Y_op = qforte.QuantumOperator()
            y_circ = qforte.QuantumCircuit()

            x_circ.add_gate(qforte.make_gate('X', ancilla_idx, ancilla_idx))
            y_circ.add_gate(qforte.make_gate('Y', ancilla_idx, ancilla_idx))

            X_op.add_term(1.0, x_circ)
            Y_op.add_term(1.0, y_circ)

            # Initalize experiment
            X_exp = qforte.Experiment(nqubits+1, cir, X_op, 100)
            Y_exp = qforte.Experiment(nqubits+1, cir, Y_op, 100)

            params = [1.0]
            x_value = X_exp.perfect_experimental_avg(params)
            y_value = Y_exp.perfect_experimental_avg(params)


            # <Psi_o|Ub V_l Uk|Psi_o> = <2* sigma_+> = <Psi_f|X|Psi_f> + * i * <Psi_f|Y|Psi_f>
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

    #Initialize arrays for Hbar and S
    h_mat = np.zeros((num_tot_basis,num_tot_basis), dtype=complex)
    s_mat = np.zeros((num_tot_basis,num_tot_basis), dtype=complex)

    #Build |Ωn> and |H Ωn> Vectors
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

# def update_mr_mats_fast(s_mat, h_mat, ref_lst, nstates_per_ref, dt_lst, H, nqubits, trot_order=1):
#     """
#     This function updates the S and H matricies with a new reference and guesses a new one.
#
#     :param ref: a list representing the referende state |Phi_o>
#     :param dt: a double representing the real time step
#     :param H: the QuantumOperator to time evolove under
#     :param nqubits: the intager number of qubits
#     :param A: (optional) the overal operator to measure with respect to
#     """
#     start_idx = (len(ref_lst)-1) * nstates_per_ref
#     end_idx = len(ref_lst) * nstates_per_ref
#     # num_tot_basis = len(ref_lst) * nstates_per_ref
#
#     #Initialize arrays for Hbar and S
#     # h_mat = np.zeros((num_tot_basis,num_tot_basis), dtype=complex)
#     # s_mat = np.zeros((num_tot_basis,num_tot_basis), dtype=complex)
#
#     #Build |Ωn> and |H Ωn> Vectors
#     omega_lst = []
#     Homega_lst = []
#
#     for i, ref in enumerate(ref_lst):
#         dt = dt_lst[i]
#         for n in range(nstates_per_ref):
#
#             Un = qforte.QuantumCircuit()
#             for j in range(nqubits):
#                 if ref[j] == 1:
#                     Un.add_gate(qforte.make_gate('X', j, j))
#                     phase1 = 1.0
#
#             if(n>0):
#                 temp_op1 = qforte.QuantumOperator()
#                 for t in H.terms():
#                     c, op = t
#                     phase = -1.0j * n * dt * c
#                     temp_op1.add_term(phase, op)
#
#                 expn_op1, phase1 = qforte.trotterization.trotterize(temp_op1, trotter_order=trot_order)
#                 Un.add_circuit(expn_op1)
#
#             QC = qforte.QuantumComputer(nqubits)
#             QC.apply_circuit(Un)
#             QC.apply_constant(phase1)
#             omega_lst.append(np.asarray(QC.get_coeff_vec(), dtype=complex))
#
#             Homega = np.zeros((2**nqubits), dtype=complex)
#
#             for k in range(len(H.terms())):
#                 QCk = qforte.QuantumComputer(nqubits)
#                 QCk.set_coeff_vec(QC.get_coeff_vec())
#
#                 if(H.terms()[k][1] is not None):
#                     QCk.apply_circuit(H.terms()[k][1])
#                 if(H.terms()[k][0] is not None):
#                     QCk.apply_constant(H.terms()[k][0])
#
#                 Homega = np.add(Homega, np.asarray(QCk.get_coeff_vec(), dtype=complex))
#
#             Homega_lst.append(Homega)
#
#     for p in range(num_tot_basis):
#         for q in range(p, num_tot_basis):
#             h_mat[p][q] = np.vdot(omega_lst[p], Homega_lst[q])
#             h_mat[q][p] = np.conj(h_mat[p][q])
#             s_mat[p][q] = np.vdot(omega_lst[p], omega_lst[q])
#             s_mat[q][p] = np.conj(s_mat[p][q])
#
#     return s_mat, h_mat

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

    #Initialize arrays for Hbar and S
    h_mat = np.zeros((num_tot_basis,num_tot_basis), dtype=complex)
    s_mat = np.zeros((num_tot_basis,num_tot_basis), dtype=complex)

    #Build |Ωn> and |H Ωn> Vectors
    omega_lst = []
    Homega_lst = []

    print('\nref_list:')
    print(ref_lst)

    for i, ref in enumerate(ref_lst):
        dt = dt_lst[i]
        for n in range(nstates_per_ref):

            Un = qforte.QuantumCircuit()
            # for j in range(nqubits):
            #     if ref[j] == 1:
            #         Un.add_gate(qforte.make_gate('X', j, j))

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

            ############################################
            # put QC in referece state without using gates...
            state_prep_lst = []
            for term in ref:
                coeff = term[0]
                det = term[1]
                # print('det: ', det)
                # print('coeff: ', coeff)
                idx = ref_to_basis_idx(det)
                state = qforte.QuantumBasis(idx)
                state_prep_lst.append( (state, coeff) )


            ############################################
            QC.set_state(state_prep_lst)
            # print('\n')
            # qforte.smart_print(QC)
            QC.apply_circuit(Un)
            QC.apply_constant(phase1)
            omega_lst.append(np.asarray(QC.get_coeff_vec(), dtype=complex))

            Homega = np.zeros((2**nqubits), dtype=complex)

            # print('\n')
            # qforte.smart_print(QC)

            # counter = 0
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

    print('\n\ns is of inital rank:    ', len(s))
    print('\n\ns is of truncated rank: ', len(s_prime))

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
