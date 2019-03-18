"""
Functions for exponentiation of qubit operator terms (circuits)
"""

import qforte
import numpy


def exponentiate_single_term(param, term):
    """
    A function which returns an exponentiated version of a single (param * term)
    from a parameterized qubit state preparation circuit.

    :param term: the term to be exponentiated
    """

    # TODO: look into multiplication by 1.0j? (Nick)
    # TODO: This code is very similar to that used in PyQuil,
    # make sure this is ok (Nick)

    if not numpy.isclose(numpy.imag(param), 0.0):
        param *= 1.0j
        print('warning: term had imaginary parameter so multipled by i')

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

        if ('X' == id):
            to_z.add_gate(qforte.make_gate('H', target, control))
            to_original.add_gate(qforte.make_gate('H', target, control))
        elif ('Y' == id):
            to_z.add_gate(qforte.make_gate('Rx', target, control, numpy.pi/2.0))
            to_original.add_gate(qforte.make_gate('Rx', target, control, -numpy.pi/2.0))
        elif ('I' == id):
            continue

        if (prev_target is not None):
            cX_circ.add_gate(qforte.make_gate('cX', target, prev_target))

        prev_target = target
        max_target = target

    #gate that actually contains the parameterization for the term
    z_rot = qforte.make_gate('Rz', max_target, max_target, 2.0*numpy.real(param))
    cX_circ.set_reversed_gates();

    # qforte.smart_print(to_z)
    # qforte.smart_print(to_original)
    # qforte.smart_print(cX_circ)

    #assemble the actual exponential
    for gate in to_z.gates():
        exponential.add_gate(gate)
    for gate in cX_circ.gates():
        exponential.add_gate(gate)

    exponential.add_gate(z_rot)

    for gate in cX_circ.reversed_gates():
        exponential.add_gate(gate)
    for gate in to_original.gates():
        exponential.add_gate(gate)

    return exponential
