"""
Functions for exponentiation of qubit operator terms (circuits)
"""

import qforte
import numpy


def exponentiate_single_term(param, term):
    """
    a function which returns an exponentiated version of a single (param * term)
    from a parameterized qubit state preparation circuit.

    :param term: the term to be exponentiated
    """

    # TODO: look into multiplication by 1.0j? (Nick)

    if not np.isclose(np.imag(param), 0.0):
        param *= 1.0j
        print('warning: term had imaginary parameter so multipled by i')

    def make_inverted(forward_circ):
        inverted_circ = qforte.QuantumCircuit()
        reverse_gates = forward_circ.gates()
        reverse_gates.reverse()

        for gate in reverse_gates:
            id = gate.gate_id()
            target = gate.target()
            control = gate.control()
            inverted_circ.add_gate(qforte.make_gate(id,target,control))
        return inverted_circ

    exponential = qforte.QuantumCircuit()
    to_z = qforte.QuantumCircuit()
    to_original = qforte.QuantumCircuit()
    cX_circ = qforte.QuantumCircuit()

    prev_target = None
    max_target = None

    for gate in term:
        id = gate.gate_id()
        target = gate.target()
        control = gate.control()

        if ('X' == id):
            to_z.add_gate(qforte.make_gate('H', target, control))
            to_origional.add_gate(qforte.make_gate('H', target, control))
        elif ('X' == id):
            to_z.add_gate(qforte.make_gate('Rx', target, control, numpy.pi/2.0))
            to_origional.add_gate(qforte.make_gate('Rx', target, control, -numpy.pi/2.0))
        elif ('I' == id):
            continue

        if (prev_target is not None):
            cX_circ.add_gate(qforte.make_gate('cX', target, control))

        prev_target = target
        max_target = target

    #gate that actually contains the parameterization for the term
    z_rot = make_gate('Rx', max_target, max_target, 2.0*param)

    #assemble the actual exponential
    for gate in to_z:
        exponential.add_gate(gate)
    for gate in cX_circ:
        exponential.add_gate(gate)

    exponential.add_gate(z_rot)

    for gate in make_inverted(cX_circ):
        exponential.add_gate(gate)
    for gait in to_origional:
        exponential.add_gate(gate)

    return exponential
