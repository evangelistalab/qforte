"""
qpe_helpers.py
=================================================
A module containing helper functions for
the quantum phase estimation algorithm.
"""

import qforte

import numpy as np

def get_Uprep(ref, state_prep):
    """Generates a circuit whcih builds the initalization state for phase
    estimation.

        Arguments
        ---------

        ref : list
            The initial reference state given as a list of 1's and 0's
            (e.g. the Hartree-Fock state).

        state_prep : string
            The desired approach for inital state preparation. Specifying
            'single_reference' will build a circuit which will reproduce the
            porvieded ref state.

        Returns
        -------

        Uprep : QuantumCircuit
            A circuit to consturct the initiaization state.
    """
    Uprep = qforte.QuantumCircuit()
    if state_prep == 'single_reference':
        for j in range(len(ref)):
            if ref[j] == 1:
                Uprep.add_gate(qforte.make_gate('X', j, j))
    else:
        raise ValueError("Only 'single_reference' supported as state preparation type")

    return Uprep

def get_Uhad(abegin, aend):
    """Generates a circuit which to puts all of the ancilla regester in
    superpostion.

        Arguments
        ---------

        abegin : int
            The index of the begin qubit.

        aend : int
            The index of the end qubit.

        Returns
        -------

        qft_circ : QuantumCircuit
            A circuit of consecutive Hadamard gates.
    """
    Uhad = qforte.QuantumCircuit()
    for j in range(abegin, aend + 1):
        Uhad.add_gate(qforte.make_gate('H', j, j))

    return Uhad

def get_dynamics_circ(H, trotter_num, abegin, n_ancilla, t=1.0):
    """Generates a circuit for controlled dynamics operations used in phase
    estimation. It approximates the exponentiated hermetina operator H as e^-iHt.

        Arguments
        ---------

        H : QuantumOperator
            The hermetian operaotr whos dynamics and eigenstates are of interest,
            ususally the Hamiltonian.

        trotter_num : int
            The trotter number (m) to use for the decompostion. Exponentiation
            is exact in the m --> infinity limit.

        abegin : int
            The index of the begin qubit.

        n_ancilla : int
            The number of anciall qubit used for the phase estimation.
            Determintes the total number of time steps.

        t : float
            The total evolution time.

        Returns
        -------

        Udyn : QuantumCircuit
            A circuit approximating controlled application of e^-iHt.
    """
    Udyn = qforte.QuantumCircuit()
    ancilla_idx = abegin
    total_phase = 1.0
    for n in range(n_ancilla):
        tn = 2 ** n
        temp_op = qforte.QuantumOperator()
        scaler_terms = []
        for h in H.terms():
            c, op = h
            phase = -1.0j * t * c #* tn
            temp_op.add_term(phase, op)
            gates = op.gates()
            if op.size() == 0:
                scaler_terms.append(c * t)


        expn_op, phase1 = qforte.trotterization.trotterize_w_cRz(temp_op,
                                                                 ancilla_idx,
                                                                 trotter_number=trotter_num)

        Udyn.add_gate(qforte.make_gate('R', ancilla_idx, ancilla_idx,  -1.0 * np.sum(scaler_terms) * float(tn)))

        # TODO: see if the below expression works with a multiplier...
        for i in range(tn):
            for gate in expn_op.gates():
                Udyn.add_gate(gate)

        ancilla_idx += 1

    return Udyn


def qft_circuit(abegin, aend, direct):
    """Generates a circuit for Quantum Fourier Transformation with no swaping
    of bit positions.

        Arguments
        ---------

        abegin : int
            The index of the begin qubit.

        aend : int
            The index of the end qubit.

        direct : string
            The direction of the Fourier Transform can be 'forward' or 'reverse.'

        Returns
        -------

        qft_circ : QuantumCircuit
            A circuit representing the Quantum Fourier Transform.
    """

    qft_circ = qforte.QuantumCircuit()
    lens = aend - abegin + 1
    for j in range(lens):
        qft_circ.add_gate(qforte.make_gate('H', j+abegin, j+abegin))
        for k in range(2, lens+1-j):
            phase = 2.0*np.pi/(2**k)
            qft_circ.add_gate(qforte.make_gate('cR', j+abegin, j+k-1+abegin, phase))

    if direct == 'forward':
        return qft_circ
    elif direct == 'reverse':
        return qft_circ.adjoint()
    else:
        raise ValueError('QFT directions can only be "forward" or "reverse"')

    return qft_circ

def get_z_circuit(abegin, aend):
    """Generates a circuit of Z gates for each quibit in the ancilla register.

        Arguments
        ---------

        abegin : int
            The index of the begin qubit.

        aend : int
            The index of the end qubit.

        Returns
        -------

        z_circ : QuantumCircuit
            A circuit representing the the Z gates to be measured.
    """

    Z_circ = qforte.QuantumCircuit()
    for j in range(abegin, aend + 1):
        Z_circ.add_gate(qforte.make_gate('Z', j, j))

    return Z_circ
