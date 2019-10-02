import qforte
# from qforte.utils import transforms
from qforte.utils import trotterization as trot

import numpy as np
# from scipy import linalg

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

def matrix_element(ref, dt, m, n, H, nqubits, A = None):
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
    expn_op1, phase1 = qforte.trotterization.trotterize_w_cRz(temp_op1, ancilla_idx)

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
    expn_op2, phase2 = qforte.trotterization.trotterize_w_cRz(temp_op2, ancilla_idx, Use_open_cRz=False)

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

def matrix_element_fast(ref, dt, m, n, H, nqubits, A = None):
    """
    This functio returns a single matrix element M_bk based on the evolutio of
    two unitary operators Ub = exp(-i * m * dt * H) and H_q = exp(-i * n * dt *H) on a
    reference state |Phi_o>. This is done WITHOUT measuring any operators,
    but rather computes the expecation value directly using a priori knowlege of
    the wavefunction coefficients

    :param ref: a list representing the referende state |Phi_o>
    :param dt: a double representing the real time step
    :param m: the intager number of time steps for the Ub evolution
    :param n: the intager number of time steps for the Uk evolution
    :param H: the QuantumOperator to time evolove under
    :param nqubits: the intager number of qubits
    :param A: (optional) the overal operator to measure with respect to
    """
    value = 0.0

    # Prepare the right circuit exp(-i n dt H) prod_j X_j
    Uk = qforte.QuantumCircuit()
    # 1. Add all the X gates (proj_j X_j) that define the reference
    for j in range(nqubits):
        if ref[j] == 1:
            Uk.add_gate(qforte.make_gate('X', j, j))

    # 2. prod_l exp(-i n dt h_l P_l)
    temp_op1 = qforte.QuantumOperator() # A temporary operator to multiply H by
    for t in H.terms():
        c, op = t
        phase = -1.0j * n * dt * c
        temp_op1.add_term(phase, op)

    expn_op1, phase1 = qforte.trotterization.trotterize(temp_op1)

    for gate in expn_op1.gates():
        Uk.add_gate(gate)

    # Prepare the left circuit exp(-i n dt H) prod_j X_j
    Ub = qforte.QuantumCircuit()
    # 1. rev_prod_k exp(i n dt h_k P_k)
    temp_op2 = qforte.QuantumOperator()
    for t in reversed(H.terms()):
        c, op = t
        phase = 1.0j * m * dt * c
        temp_op2.add_term(phase, op)

    expn_op2, phase2 = qforte.trotterization.trotterize(temp_op2)

    for gate in expn_op2.gates():
        Ub.add_gate(gate)

    # 2. Add all the X gates that define the reference
    for j in range(nqubits):
        if ref[j] == 1:
            Ub.add_gate(qforte.make_gate('X', j, j))

    if A == None:
        cir = qforte.QuantumCircuit()
        cir.add_circuit(Uk)
        cir.add_circuit(Ub)

        # Projection approach <0| (XPX |0>)
        zero_state = qforte.QuantumBasis()
        qc = qforte.QuantumComputer(nqubits)
        qc.apply_circuit(cir)
        value = qc.coeff(zero_state) * phase1 * phase2

    else:
        for t in A.terms():
            c, op = t
            cir = qforte.QuantumCircuit()
            cir.add_circuit(Uk)
            cir.add_circuit(op)
            cir.add_circuit(Ub)

            # Projection approach <0| (XPX |0>)
            zero_state = qforte.QuantumBasis()
            qc = qforte.QuantumComputer(nqubits)
            qc.apply_circuit(cir)
            element = qc.coeff(zero_state) * phase1 * phase2
            value += c * element

            #append to use measurement

    return value

def mr_matrix_element(ref_I, ref_J, dt_I, dt_J, m, n, H, nqubits, A = None):
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
    expn_op1, phase1 = qforte.trotterization.trotterize_w_cRz(temp_op1, ancilla_idx)
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
    expn_op2, phase2 = qforte.trotterization.trotterize_w_cRz(temp_op2, ancilla_idx, Use_open_cRz=False)
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

def mr_matrix_element_fast(ref_I, ref_J, dt_I, dt_J, m, n, H, nqubits, A = None):
    """
    This functio returns a single matrix element M_bk based on the evolutio of
    two unitary operators Ub = exp(-i * m * dt * H) and H_q = exp(-i * n * dt *H) on a
    reference state |Phi_o>. This is done WITHOUT measuring any operators,
    but rather computes the expecation value directly using a priori knowlege of
    the wavefunction coefficients

    :param ref: a list representing the referende state |Phi_o>
    :param dt: a double representing the real time step
    :param m: the intager number of time steps for the Ub evolution
    :param n: the intager number of time steps for the Uk evolution
    :param H: the QuantumOperator to time evolove under
    :param nqubits: the intager number of qubits
    :param A: (optional) the overal operator to measure with respect to
    """
    value = 0.0

    # Prepare the right circuit exp(-i n dt H) prod_j X_j
    Uk = qforte.QuantumCircuit()
    # 1. Add all the X gates (proj_j X_j) that define the reference
    for j in range(nqubits):
        if ref_I[j] == 1:
            Uk.add_gate(qforte.make_gate('X', j, j))

    # 2. prod_l exp(-i n dt h_l P_l)
    temp_op1 = qforte.QuantumOperator() # A temporary operator to multiply H by
    for t in H.terms():
        c, op = t
        phase = -1.0j * m * dt_I * c
        temp_op1.add_term(phase, op)

    expn_op1, phase1 = qforte.trotterization.trotterize(temp_op1)

    for gate in expn_op1.gates():
        Uk.add_gate(gate)

    # Prepare the left circuit exp(-i n dt H) prod_j X_j
    Ub = qforte.QuantumCircuit()
    # 1. rev_prod_k exp(i n dt h_k P_k)
    temp_op2 = qforte.QuantumOperator()
    for t in reversed(H.terms()):
        c, op = t
        phase = 1.0j * n * dt_J * c
        temp_op2.add_term(phase, op)

    expn_op2, phase2 = qforte.trotterization.trotterize(temp_op2)

    for gate in expn_op2.gates():
        Ub.add_gate(gate)

    # 2. Add all the X gates that define the reference
    for j in range(nqubits):
        if ref_J[j] == 1:
            Ub.add_gate(qforte.make_gate('X', j, j))

    if A == None:
        cir = qforte.QuantumCircuit()
        cir.add_circuit(Uk)
        cir.add_circuit(Ub)

        # Projection approach <0| (XPX |0>)
        zero_state = qforte.QuantumBasis()
        qc = qforte.QuantumComputer(nqubits)
        qc.apply_circuit(cir)
        value = qc.coeff(zero_state) * phase1 * phase2

    else:
        for t in A.terms():
            c, op = t
            cir = qforte.QuantumCircuit()
            cir.add_circuit(Uk)
            cir.add_circuit(op)
            cir.add_circuit(Ub)

            # Projection approach <0| (XPX |0>)
            zero_state = qforte.QuantumBasis()
            qc = qforte.QuantumComputer(nqubits)
            qc.apply_circuit(cir)
            element = qc.coeff(zero_state) * phase1 * phase2
            value += c * element

    return value
