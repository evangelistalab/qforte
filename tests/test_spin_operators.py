from pytest import approx
import qforte as qf
import numpy as np


class TestSpinOperators:
    def test_spin_operators(self):
        # In this test, we create four spin states, namely,
        # |S = 0, Sz=0>  = 1/np.sqrt(2) (|1001> - |0110>)
        # |S = 1, Sz=-1> = |0101>
        # |S = 1, Sz=0>  = 1/np.sqrt(2) (|1001> + |0110>)
        # |S = 1, Sz=1>  = |1010>
        # and compute <S_z> and <S^2>
        # To test S_plus and S_minus we have them act on the
        # above states and then compute <S_z> and <S^2>
        # Note that the spin-orbitals in QForte follow the
        # alpha,beta,alpha,beta,... ordering.

        # Define required determinants
        basis_1010 = qf.QubitBasis(5)
        basis_0110 = qf.QubitBasis(6)
        basis_1001 = qf.QubitBasis(9)
        basis_0101 = qf.QubitBasis(10)

        # Initialize quantum computers in the |0000>
        singlet = qf.Computer(4)
        triplet_minus_1 = qf.Computer(4)
        triplet_0 = qf.Computer(4)
        triplet_plus_1 = qf.Computer(4)

        # Bring the quantum computers to the desired spin states
        singlet.set_state([(basis_1001, 1 / np.sqrt(2)), (basis_0110, -1 / np.sqrt(2))])
        triplet_minus_1.set_state([(basis_0101, 1)])
        triplet_0.set_state(
            [(basis_1001, 1 / np.sqrt(2)), (basis_0110, 1 / np.sqrt(2))]
        )
        triplet_plus_1.set_state([(basis_1010, 1)])

        # Define spin operators
        Sz = qf.total_spin_z(4)
        S_plus = qf.total_spin_raising(4)
        S_minus = qf.total_spin_lowering(4)
        S_squared = qf.total_spin_squared(4)

        results = []

        for state in [singlet, triplet_minus_1, triplet_0, triplet_plus_1]:
            for spin_operator in [S_plus, S_minus, Sz, S_squared]:
                if spin_operator in [S_plus, S_minus]:
                    temp = qf.Computer(4)
                    temp.set_coeff_vec(state.get_coeff_vec())
                    temp.apply_operator(spin_operator)
                    # the states resulting from applications of S_plus
                    # and S_minus require normalization
                    coeff_vec = temp.get_coeff_vec()
                    if all(i == 0 for i in coeff_vec):
                        results.append("null")
                        continue
                    normalize = np.sqrt(
                        sum(map(lambda i: i * i.conjugate(), coeff_vec))
                    )
                    coeff_vec_normalized = [i / normalize for i in coeff_vec]
                    temp.set_coeff_vec(coeff_vec_normalized)
                    results.append(temp.direct_op_exp_val(Sz))
                else:
                    results.append(state.direct_op_exp_val(spin_operator))

        # A 'null' means that S_plus/S_minus annihilates the state
        assert results[0] == "null"
        assert results[1] == "null"
        assert results[2] == approx(0, abs=1.0e-14)
        assert results[3] == approx(0, abs=1.0e-14)
        assert results[4] == approx(0, abs=1.0e-14)
        assert results[5] == "null"
        assert results[6] == approx(-1, abs=1.0e-14)
        assert results[7] == approx(2, abs=1.0e-14)
        assert results[8] == approx(1, abs=1.0e-14)
        assert results[9] == approx(-1, abs=1.0e-14)
        assert results[10] == approx(0, abs=1.0e-14)
        assert results[11] == approx(2, abs=1.0e-14)
        assert results[12] == "null"
        assert results[13] == approx(0, abs=1.0e-14)
        assert results[14] == approx(1, abs=1.0e-14)
        assert results[15] == approx(2, abs=1.0e-14)
