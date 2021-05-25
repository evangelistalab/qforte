"""
Functions for exponentiation of qubit operator terms (circuits)
"""

import qforte
import numpy as np

def exponentiate_single_term(coefficient, term, Use_cRz=False, ancilla_idx=None, Use_open_cRz=False):
    """
    returns the exponential of an string of Pauli operators multiplied by an imaginary coefficient

        exp(coefficient * term)

    Parameters
    ----------
    :param coefficient: float
        an imaginary coefficient that multiplies the Pauli string
    :param term: QuantumCircuit
        a Pauli string to be exponentiated
    """
    # This function assumes that the factor is imaginary. The following tests for it.
    if np.abs(np.real(coefficient)) > 1.0e-16:
        print("exp factor: ", coefficient)
        raise ValueError('exponentiate_single_term() called with a real coefficient')

    # If the Pauli string has no terms this is just a phase factor
    if term.size() == 0:
        return (qforte.QuantumCircuit(), np.exp(coefficient))

    exponential = qforte.QuantumCircuit()
    to_z = qforte.QuantumCircuit()
    to_original = qforte.QuantumCircuit()
    cX_circ = qforte.QuantumCircuit()

    prev_target = None
    max_target = None

    for gate in term.gates():
        id = gate.gate_id()
        target = gate.target()
        control = gate.control()

        if (id == 'X'):
            to_z.add(qforte.gate('H', target, control))
            to_original.add(qforte.gate('H', target, control))
        elif (id == 'Y'):
            to_z.add(qforte.gate('Rzy', target, control))
            to_original.add(qforte.gate('Rzy', target, control))
        elif (id == 'I'):
            continue

        if (prev_target is not None):
            cX_circ.add(qforte.gate('cX', target, prev_target))

        prev_target = target
        max_target = target

    # Gate that actually contains the parameterization for the term
    # TODO(Nick): investigate real/imaginary usage of 'factor' in below expression

    if(Use_cRz):
        z_rot = qforte.gate('cRz', max_target, ancilla_idx, -2.0 * np.imag(coefficient))
    else:
        z_rot = qforte.gate('Rz', max_target, max_target, -2.0 * np.imag(coefficient))

    # Assemble the actual exponential
    exponential.add(to_z)
    exponential.add(cX_circ)

    if(Use_open_cRz):
        exponential.add(qforte.gate('X', ancilla_idx, ancilla_idx))

    exponential.add(z_rot)

    if(Use_open_cRz):
        exponential.add(qforte.gate('X', ancilla_idx, ancilla_idx))

    adj_cX_circ = cX_circ.adjoint()
    exponential.add(adj_cX_circ)
    adj_to_z = to_z.adjoint()
    exponential.add(adj_to_z)

    return (exponential, 1.0)
