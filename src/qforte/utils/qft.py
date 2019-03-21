import qforte
import numpy

def qft_circuit(n):

    """
    qft_circuit is a function that generates circuit for
    Quantum Fourier Transformation implemented in a efficient
    approach.

    :param n: the number of qubits
    """

    # Build qft circuit
    qft_circ = qforte.QuantumCircuit()
    for j in range(n):
        qft_circ.add_gate(qforte.make_gate('H', j, j))
        for k in range(2, n+1-j):
            phase = 2.0*numpy.pi/(2**k)
            qft_circ.add_gate(qforte.make_gate('cR', j, j+k-1, phase))

    # Build reversing circuit
    if n % 2 == 0:
        for i in range(int(n/2)):
            qft_circ.add_gate(qforte.make_gate('SWAP', i, n-1-i))
    else:
        for i in range(int((n-1)/2)):
            qft_circ.add_gate(qforte.make_gate('SWAP', i, n-1-i))

    return qft_circ

def qft(qc_state, n):

    """
    qft is a function that performs a Quantum Fourier
    Transformation on QuantumComputer states

    :param qc_state: the input QuantumComputer state
    """

    if isinstance(qc_state, qforte.QuantumComputer):
        print('')
    else:
        return NotImplemented

    # Apply qft circuits
    circ = qft_circuit(n)
    qc_state.apply_circuit(circ)

    # Normalize coeffs
    coeff_ = qc_state.get_coeff_vec()
    for a in coeff_:
        a *= 1.0/numpy.sqrt(2)

    return qc_state

def rev_qft(qc_state, n):

    """
    qft is a function that performs a inversed Quantum 
    Fourier Transformation on QuantumComputer states

    :param qc_state: the input QuantumComputer state
    """

    if isinstance(qc_state, qforte.QuantumComputer):
        print('')
    else:
        return NotImplemented
    
    # Apply qft circuits
    circ = qft_circuit(n)
    circ.reversed_gates()
    qc_state.apply_circuit(circ)

    # Normalize coeffs
    coeff_ = qc_state.get_coeff_vec()
    for a in coeff_:
        a *= 1.0/numpy.sqrt(2)

    return qc_state          
