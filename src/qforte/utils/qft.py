import qforte
import numpy


def qft_circuit(na, nb, direct):
    """
    generates a circuit for Quantum Fourier Transformation

    :param na: (int) the begin qubit
    :param nb: (int) the end qubit

    :param direct: (string) the direction of the Fourier Transform
    can be 'forward' or 'reverse'
    """

    # Build qft circuit
    qft_circ = qforte.Circuit()
    lens = nb - na + 1
    for j in range(lens):
        qft_circ.add(qforte.gate("H", j + na, j + na))
        for k in range(2, lens + 1 - j):
            phase = 2.0 * numpy.pi / (2**k)
            qft_circ.add(qforte.gate("cR", j + na, j + k - 1 + na, phase))

    # Build reversing circuit
    if lens % 2 == 0:
        for i in range(int(lens / 2)):
            qft_circ.add(qforte.gate("SWAP", i + na, lens - 1 - i + na))
    else:
        for i in range(int((lens - 1) / 2)):
            qft_circ.add(qforte.gate("SWAP", i + na, lens - 1 - i + na))

    if direct == "forward":
        return qft_circ
    elif direct == "reverse":
        return qft_circ.adjoint()
    else:
        raise ValueError('QFT directions can only be "forward" or "reverse"')

    return qft_circ


def qft(qc_state, na, nb):
    """
    performs a Quantum Fourier Transformation on Computer states

    :param qc_state: (Computer) the input Computer state
    :param na: (int) the begin qubit
    :param nb: (int) the end qubit

    """

    if not isinstance(qc_state, qforte.Computer):
        return NotImplemented

    # Apply qft circuits
    circ = qft_circuit(na, nb, "forward")
    qc_state.apply_circuit(circ)

    # Normalize coeffs
    coeff_ = qc_state.get_coeff_vec()
    for a in coeff_:
        a *= 1.0 / numpy.sqrt(2)

    return qc_state


def rev_qft(qc_state, na, nb):
    """
    performs a inversed QuantumFourier Transformation on Computer states

    :param qc_state: (Computer) the input Computer
    :param na: (int) the begin qubit
    :param nb: (int) the end qubit

    """

    if not isinstance(qc_state, qforte.Computer):
        return NotImplemented

    # Apply qft circuits
    circ = qft_circuit(na, nb, "reverse")
    adj_circ = circ.adjoint()
    qc_state.apply_circuit(adj_circ)

    # Normalize coeffs
    coeff_ = qc_state.get_coeff_vec()
    for a in coeff_:
        a *= 1.0 / numpy.sqrt(2)

    return qc_state
