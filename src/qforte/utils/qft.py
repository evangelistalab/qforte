import qforte
import numpy

"""
generates a circuit for Quantum Fourier Transformation

:param n: (int) the number of qubits

:param direct: (string) the direction of the Fourier Transform
can be 'forward' or 'reverse'
"""

def qft_circuit(n, direct):

    # Build qft circuit
    qft_circ = qforte.QuantumCircuit()
    for j in range(n):
        qft_circ.add_gate(qforte.make_gate('H', j, j))
        for k in range(2, n+1-j):
            phase = 2.0*numpy.pi/(2**k)
            if direct == 'forward':
                qft_circ.add_gate(qforte.make_gate('cR', j, j+k-1, phase))
            if direct == 'reverse':
                qft_circ.add_gate(qforte.make_gate('cR', j, j+k-1, (-1)*phase))

    # Build reversing circuit
    if n % 2 == 0:
        for i in range(int(n/2)):
            qft_circ.add_gate(qforte.make_gate('SWAP', i, n-1-i))
    else:
        for i in range(int((n-1)/2)):
            qft_circ.add_gate(qforte.make_gate('SWAP', i, n-1-i))

    return qft_circ

"""
performs a Quantum Fourier Transformation on QuantumComputer states

:param qc_state: (QuantumComputer) the input QuantumComputer state

:param n: (int) the number of qubits
"""

def qft(qc_state, n):

    if isinstance(qc_state, qforte.QuantumComputer):
        print('')
    else:
        return NotImplemented

    # Apply qft circuits
    circ = qft_circuit(n, 'forward')
    qc_state.apply_circuit(circ)

    # Normalize coeffs
    coeff_ = qc_state.get_coeff_vec()
    for a in coeff_:
        a *= 1.0/numpy.sqrt(2)

    return qc_state

"""
performs a inversed QuantumFourier Transformation on QuantumComputer states

:param qc_state: (QuantumComputer) the input QuantumComputer

:param n: (int) the number of qubits
"""

def rev_qft(qc_state, n):

    if isinstance(qc_state, qforte.QuantumComputer):
        print('')
    else:
        return NotImplemented

    # Apply qft circuits
    circ = qft_circuit(n, 'reverse')
    circ.reversed_gates()
    qc_state.apply_circuit(circ)

    # Normalize coeffs
    coeff_ = qc_state.get_coeff_vec()
    for a in coeff_:
        a *= 1.0/numpy.sqrt(2)

    return qc_state
