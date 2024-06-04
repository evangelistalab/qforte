"""
Functions for constructing compact quantum circuits for fermioninc/qubit excitations
"""

import qforte as qf
import numpy as np


def compact_excitation_circuit(theta, creation, annihilation, qubit_excitations):
    """
    This function constructs compact quantum circuits for fermionic/qubit
    excitations of the form
    exp(theta * κ_μ),
    where κ_μ is a second quantized (κ_μ = t_μ - t_μ^\dagger) or
    qubit (κ_μ = Q_μ - Q_μ^\dagger) excitation operator.
    The resulting quantum circuits have the minimum (so far) number
    of CNOTs. The original idea is reported in DOI: 10.1103/PhysRevA.102.062612
    and the extension to arbitrary excitations in DOI: 10.1021/acs.jctc.2c01016
    Arguments
    =========
    theta: float
        Excitation amplitude.
    creation: list of ints
        Indices of creation operators in the excitation component.
    annihilation: list of ints
        Indices of annihilation operators in the excitation component.
    qubit_excitations: bool
        Controls the use of qubit vs fermionic excitations.
    Returns
    =======
    circ: Circuit
        Quantum circuit of fermionic/qubit excitation.
    """

    if len(creation) != len(annihilation):
        raise ValueError(
            "Compact fermionic/qubit excitations are implemented for particle-number-conserving operators only."
        )

    # When using GSD, there are excitations of the form p^ q^ s q, etc.
    # Such excitations can be viewed as controled single excitations,
    # and require special treatment.
    creation_unique = [x for x in creation if x not in annihilation]
    annihilation_unique = [x for x in annihilation if x not in creation]
    gsd_control = list(set(creation) & set(annihilation))

    if len(gsd_control) > 1:
        # Excitations of the form p^ q^ p q are not included in the GSD pool of QForte
        raise ValueError("A GSD excitation of the form p^ q^ p q was encountered!")

    gsd_sign = 1
    gsd_unique = False
    if gsd_control:
        gsd_sign = -1
        # check if a CNOT staircase involving unique indicies is required
        duplicate = []
        for i in range(2):
            if creation[i] == gsd_control[0]:
                duplicate.append(i)
            if annihilation[i] == gsd_control[0]:
                duplicate.append(i)
        if duplicate == [0, 0] or duplicate == [1, 1]:
            gsd_unique = True

    circ = qf.Circuit()

    # Construct CNOT staircase associated with fermion sign
    if not qubit_excitations:
        if gsd_unique:
            CNOT_stair = fermion_sign_circuit(creation_unique, annihilation_unique)
        else:
            CNOT_stair = fermion_sign_circuit(creation, annihilation)

        circ.add(CNOT_stair)

    circ.add(
        qubit_excitation(
            theta,
            creation_unique,
            annihilation_unique,
            gsd_control,
            gsd_sign,
            qubit_excitations,
        )
    )

    # Add adjoint of CNOT staircase
    if not qubit_excitations:
        circ.add(CNOT_stair.adjoint())

    return circ


def fermion_sign_circuit(creation, annihilation):
    """
    Function that constructs the quantum circuit responsible for computing the sign
    of a given fermionic excitation. The resulting quantum circuit is comprised of
    a single CNOT staircase and a single CZ gate independent of the many-body rank of the
    second-quantized operator. The adjoint circuit is constructed in compact_excitation_circuit.
    Arguments
    =========
    creation: list of ints
        Indices of creation operators in the excitation component.
    annihilation: list of ints
        Indices of annihilation operators in the excitation component.
    Returns
    =======
    CNOT_circ: Circuit
        Quantum circuit of CNOT and CZ gates.
    """

    # number of qubits involved in excitation
    n_qubit = max(creation + annihilation) + 1

    # A 1 in a given column indicates a Z gate for this qubit.
    aux = np.zeros((n_qubit, len(creation) + len(annihilation) + 1), dtype=bool)
    for i, create in enumerate(creation):
        for j in range(create):
            aux[j, i] = 1
    for i, annihilate in enumerate(annihilation):
        for j in range(annihilate):
            aux[j, i + len(creation)] = 1

    # Find the Z operators that survive after taking products of operators
    for i in range(len(creation) + len(annihilation)):
        aux[:, -1] ^= aux[:, i]

    # Remove Z operators that multiply with X/Y
    for i in creation + annihilation:
        aux[i, -1] = 0

    CNOT_indices = []
    for idx, boolean in enumerate(aux[:, -1]):
        if boolean:
            CNOT_indices.append(idx)

    CNOT_indices.reverse()

    CNOT_circ = qf.Circuit()

    if CNOT_indices == []:
        return CNOT_circ

    for i in range(len(CNOT_indices) - 1):
        CNOT_circ.add(qf.gate("CNOT", CNOT_indices[i + 1], CNOT_indices[i]))

    CNOT_circ.add(qf.gate("cZ", creation[0], CNOT_indices[-1]))

    return CNOT_circ


def qubit_excitation(
    theta, creation, annihilation, gsd_control, gsd_sign, qubit_excitations
):
    """
    Function that performs a "qubit" excitation. Note that, unless qubit_excitations=True,
    the resulting circuit is not equivalent to a pure qubit excitation since sign factors
    have been modified to adapt it for fermionic excitations.
    Arguments
    =========
    theta: float
        Excitation amplitude.
    creation: list of ints
        Indices of creation operators in the excitation component.
    annihilation: list of ints
        Indices of annihilation operators in the excitation component.
    gsd_control: list of int
        Empty list except in the case of a generalized singles and doubles
        excitation of the form p^ q^ p s, p^ q^ r p, p^ q^ q s, or p^ q^ r q,
        in which case it holds the repeated qubit index.
    gsd_sign: -1 or 1
        Equals "1" except in the case of a generalized singles and doubles
        excitation of the form p^ q^ p s, p^ q^ r p, p^ q^ q s, or p^ q^ r q,
        in which case it equals "-1".
    qubit_excitations: bool
        Controls the use of qubit vs fermionic excitations.
    Returns
    =======
    circ: Circuit
        Quantum circuit of "qubit" excitation.
    """

    circ = qf.Circuit()

    for target in creation[1:]:
        circ.add(qf.gate("CNOT", target, creation[0]))
    for target in annihilation[1:]:
        circ.add(qf.gate("CNOT", target, annihilation[0]))
    circ.add(qf.gate("CNOT", annihilation[0], creation[0]))

    CNOT_circ_adjoint = circ.adjoint()

    circ.add(
        multi_qubit_controlled_Ry(
            theta,
            creation[0],
            creation[1:],
            annihilation,
            gsd_control,
            gsd_sign,
            qubit_excitations,
        )
    )

    circ.add(CNOT_circ_adjoint)

    return circ


def multi_qubit_controlled_Ry(
    theta,
    target,
    control_creation,
    control_annihilation,
    gsd_control,
    gsd_sign,
    qubit_excitations,
):
    """
    Function that constructs a multi-qubit-controlled Ry gate as a series of
    single-qubit Ry rotations and two-qubit CNOT and aCNOT gates.
    Arguments
    =========
    theta: float
        Excitation amplitude.
    target: int
        Index of qubit that multi-qubit-controlled Ry gate acts on.
    control_creation: list of ints
        Indices of creation operators that control multi-qubit-controlled Ry gate.
    control_annihilation: list of ints
        Indices of annihilation operators that control multi-qubit-controlled Ry gate.
    gsd_control: list of int
        Empty list except in the case of a generalized singles and doubles
        excitation of the form p^ q^ p s, p^ q^ r p, p^ q^ q s, or p^ q^ r q,
        in which case it holds the repeated qubit index.
    gsd_sign: -1 or 1
        Equals "1" except in the case of a generalized singles and doubles
        excitation of the form p^ q^ p s, p^ q^ r p, p^ q^ q s, or p^ q^ r q,
        in which case it equals "-1".
    qubit_excitations: bool
        Controls the use of qubit vs fermionic excitations.
    Returns
    =======
    circ: Circuit
        Quantum circuit of multi-qubit-controlled Ry gate.
    """

    # For a qubit excitation Q(theta) we need a multi-qubit-controlled Ry gate of
    # angle 2*theta. The factor of 2 is taken into account when computing the
    # "new_theta" angle for the relevant single-qubit Ry gates.

    # using an ordering similar to Yordanov
    control_creation.reverse()
    control_qubits = control_annihilation + control_creation + gsd_control

    num_Ry_gates = 1 << len(control_qubits)

    # the new theta value of the single qubit Ry gates.
    # WARNING: this already takes into account that the parameter of
    # the multiqubit Ry gate is 2*theta.
    new_theta = theta / (1 << (len(control_qubits) - 1))

    circ = qf.Circuit()

    for i in range(num_Ry_gates):
        sign = 1 - 2 * (i % 2)
        if not qubit_excitations:
            # In the case of fermionic excitations, there exists a sign factor that
            # multiplies the angle theta of the multi-qubit-controlled Ry gate. The
            # sign factor depends on the many-body rank of the excitation operator.
            prefactor = -1
            rank = len(control_annihilation)
            if rank == 1:
                prefactor = 1
            if not rank % 2 and not rank % 4:
                prefactor = 1
            elif not (rank - 1) % 2 and not (rank - 1) % 4:
                prefactor = 1
            sign *= prefactor * gsd_sign
        circ.add(qf.gate("Ry", target, target, sign * new_theta))
        for j, control in enumerate(control_qubits):
            if not (i + 1) % (num_Ry_gates / (1 << j + 1)):
                if not (i + 1) % (num_Ry_gates / 2) or i == num_Ry_gates - 1:
                    circ.add(qf.gate("CNOT", target, control))
                elif gsd_control != []:
                    circ.add(qf.gate("CNOT", target, control))
                else:
                    circ.add(qf.gate("aCNOT", target, control))
                break

    return circ
