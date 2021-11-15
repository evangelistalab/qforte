from pytest import approx
from qforte import PauliString

class TestFastPauli():

    def test_pauli_string_init(self):

        pauli_gate = ['[]', '[Z0]', '[X0]', '[Y0]']

        idx = 0
        for x in [0,1]:
            for z in [0,1]:
                ps = PauliString(x, z)
                assert ps.str() == pauli_gate[idx]
                idx += 1


        ps = PauliString(0b01101111, 0b01111011)
        assert ps.str() == '[Y0 Y1 X2 Y3 Z4 Y5 Y6]'

        ps = PauliString(7, 3)
        assert ps.str() == '[Y0 Y1 X2]'

    def test_pauli_string_product(self):

        pauli_products = ['[]'  , '[Z0]', '[X0]', '[Y0]',
                          '[Z0]', '[]'  , '[Y0]', '[X0]',
                          '[X0]', '[Y0]', '[]'  , '[Z0]',
                          '[Y0]', '[X0]', '[Z0]', '[]']

        phases = [1.0+0.0j, 1.0+0.0j, 1.0+0.0j, 1.0+0.0j,
                  1.0+0.0j, 1.0+0.0j, 0.0+1.0j, 0.0-1.0j,
                  1.0+0.0j, 0.0-1.0j, 1.0+0.0j, 0.0+1.0j,
                  1.0+0.0j, 0.0+1.0j, 0.0-1.0j, 1.0+0.0j]

        idx = 0
        for x_ps1 in [0,1]:
            for z_ps1 in [0,1]:
                for x_ps2 in [0,1]:
                    for z_ps2 in [0,1]:
                        ps1 = PauliString(x_ps1, z_ps1)
                        ps2 = PauliString(x_ps2, z_ps2)
                        (phase, product_string) = ps1 * ps2
                        assert phase.real == approx(phases[idx].real, abs=1.0e-16)
                        assert phase.imag == approx(phases[idx].imag, abs=1.0e-16)
                        assert product_string.str() == pauli_products[idx]
                        idx += 1

        ps1 = PauliString(0b1110011110, 0b1011100000)
        ps2 = PauliString(0b1100101010, 0b0111110011)
        (phase, product_string) = ps1 * ps2
        assert phase.real == approx(1, abs=1.0e-14)
        assert phase.imag == approx(0, abs=1.0e-14)
        assert product_string.str() == '[Z0 Z1 X2 Y4 X5 X7 Z8 Z9]'

        ps1 = PauliString(665, 211)
        ps2 = PauliString(803, 742)
        (phase, product_string) = ps1 * ps2
        assert phase.real == approx(1, abs=1.0e-14)
        assert phase.imag == approx(0, abs=1.0e-14)
        assert product_string.str() == '[Z0 X1 Z2 X3 Y4 Y5 X7 X8 Z9]'
