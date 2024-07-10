from pytest import approx
import numpy as np
from qforte import (
    Circuit,
    build_circuit,
    Computer,
    QubitOperator,
    trotterization,
    smart_print,
    gate,
)


class TestTrotter:
    def test_trotterization(self):
        circ_vec = [Circuit(), build_circuit("Z_0")]
        coef_vec = [-1.0j * 0.5, -1.0j * -0.04544288414432624]

        # the operator to be exponentiated
        minus_iH = QubitOperator()
        for i in range(len(circ_vec)):
            minus_iH.add(coef_vec[i], circ_vec[i])

        # exponentiate the operator
        Utrot, phase = trotterization.trotterize(minus_iH)

        inital_state = np.zeros(2**4, dtype=complex)
        inital_state[3] = np.sqrt(2 / 3)
        inital_state[12] = -np.sqrt(1 / 3)

        # initalize a quantum computer with above coeficients
        # i.e. ca|1100> + cb|0011>
        qc = Computer(4)
        qc.set_coeff_vec(inital_state)

        # apply the troterized minus_iH
        qc.apply_circuit(Utrot)
        qc.apply_constant(phase)

        smart_print(qc)

        coeffs = qc.get_coeff_vec()

        assert np.real(coeffs[3]) == approx(0.6980209737879599, abs=1.0e-15)
        assert np.imag(coeffs[3]) == approx(-0.423595782342996, abs=1.0e-15)
        assert np.real(coeffs[12]) == approx(-0.5187235657531178, abs=1.0e-15)
        assert np.imag(coeffs[12]) == approx(0.25349397560041553, abs=1.0e-15)

    def test_trotterization_with_controlled_U(self):
        circ_vec = [build_circuit("Y_0 X_1"), build_circuit("X_0 Y_1")]
        coef_vec = [-1.0719145972781818j, 1.0719145972781818j]

        # the operator to be exponentiated
        minus_iH = QubitOperator()
        for i in range(len(circ_vec)):
            minus_iH.add(coef_vec[i], circ_vec[i])

        ancilla_idx = 2

        # exponentiate the operator
        Utrot, phase = trotterization.trotterize_w_cRz(minus_iH, ancilla_idx)

        # Case 1: positive control

        # initalize a quantum computer
        qc = Computer(3)

        # build HF state
        qc.apply_gate(gate("X", 0, 0))

        # put ancilla in |1> state
        qc.apply_gate(gate("X", 2, 2))

        # apply the troterized minus_iH
        qc.apply_circuit(Utrot)

        smart_print(qc)

        coeffs = qc.get_coeff_vec()

        assert coeffs[5] == approx(-0.5421829373021542, abs=1.0e-15)
        assert coeffs[6] == approx(-0.8402604730072732, abs=1.0e-15)

        # Case 2: negitive control

        # initalize a quantum computer
        qc = Computer(3)

        # build HF state
        qc.apply_gate(gate("X", 0, 0))

        # apply the troterized minus_iH
        qc.apply_circuit(Utrot)

        smart_print(qc)

        coeffs = qc.get_coeff_vec()

        assert coeffs[1] == approx(1, abs=1.0e-15)
