"""
Functions for exponentiation of qubit operator terms (circuits)
"""

import qforte
import numpy

def exponentiate_single_term(factor, term):
    """
    returns the exponential of an string of Pauli operators multiplied by an imaginary factor

        exp(factor * term)

    Parameters
    ----------
    :param factor: float
        an imaginary factor that multiplies the Pauli string
    :param term: QuantumCircuit
        a Pauli string to be exponentiated
    """
    # This function assumes that the factor is imaginary. The following tests for it.
    if numpy.real(factor) != 0.0:
        raise SystemExit('exponentiate_single_term() called with a real factor')

    # If the Pauli string has no terms this is just a phase factor
    if term.size() == 0:
        return (qforte.QuantumCircuit(), numpy.exp(factor))

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
            to_z.add_gate(qforte.make_gate('H', target, control))
            to_original.add_gate(qforte.make_gate('H', target, control))
        elif (id == 'Y'):
            to_z.add_gate(qforte.make_gate('Rzy', target, control))
            to_original.add_gate(qforte.make_gate('Rzy', target, control))
#            to_z.add_gate(qforte.make_gate('Rx', target, control, numpy.pi/2.0))
#            to_original.add_gate(qforte.make_gate('Rx', target, control, -numpy.pi/2.0))
        elif (id == 'I'):
            continue

        if (prev_target is not None):
            cX_circ.add_gate(qforte.make_gate('cX', target, prev_target))

        prev_target = target
        max_target = target

    #gate that actually contains the parameterization for the term
    z_rot = qforte.make_gate('Rz', max_target, max_target, 2.0 * numpy.imag(factor))

    #assemble the actual exponential
    exponential.add_circuit(to_z)
    exponential.add_circuit(cX_circ)
    exponential.add_gate(z_rot)

    adj_cX_circ = cX_circ.adjoint()
    exponential.add_circuit(adj_cX_circ)
    adj_to_z = to_z.adjoint()
    exponential.add_circuit(adj_to_z)

    return (exponential, 1.0)
