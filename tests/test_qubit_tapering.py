from qforte import system_factory, QubitOperator, find_Z2_symmetries, taper_operator


class TestQubitTapering:
    def test_find_z2_symmetries(self):
        ### WARNING: This test is based on comparing the qubit tapering results of QForte with
        ###          those of Qiskit. To be able to make such comparisons, one needs to take
        ###          into account the different conventions in representing Slater determinants
        ###          between the two codes. In QForte, Slater determinants are represented by
        ###          alternating alpha/beta spin orbitals. In Qiskit, all alpha spin orbitals
        ###          are placed before the beta ones.

        generators_from_qiskit = (
            "[[Z7 Z6], "
            + "[Z13 Z12 Z5 Z4], "
            + "[Z13 Z11 Z9 Z7 Z5 Z3 Z1], "
            + "[Z13 Z10 Z8 Z7 Z5 Z2 Z0]]"
        )

        sigmas_from_qiskit = "[6 4 1 0]"

        unitaries_from_qiskit = (
            "[+0.707107[Z7 Z6]\n+0.707107[X6], "
            + "+0.707107[Z13 Z12 Z5 Z4]\n+0.707107[X4], "
            + "+0.707107[Z13 Z11 Z9 Z7 Z5 Z3 Z1]\n+0.707107[X1], "
            + "+0.707107[Z13 Z10 Z8 Z7 Z5 Z2 Z0]\n+0.707107[X0]]"
        )

        to_angs = 0.529177210903

        mol = system_factory(
            system_type="molecule",
            build_type="psi4",
            basis="sto-3g",
            mol_geometry=[
                ("O", (0, 0, -0.013500 * to_angs)),
                ("H", (0, 2.2728945 * to_angs, -1.588347 * to_angs)),
                ("H", (0, -2.2728945 * to_angs, -1.588347 * to_angs)),
            ],
            symmetry="c2v",
        )

        orig_ham = QubitOperator()
        orig_ham.add_op(mol.hamiltonian)

        generators, sigmas, unitaries, unitary = find_Z2_symmetries(
            mol.hamiltonian, True, True
        )

        sign = [1, 1, 1, 1]
        tapered_ham = taper_operator(sigmas, sign, mol.hamiltonian, unitary)

        assert str(generators) == generators_from_qiskit
        assert str(sigmas) == sigmas_from_qiskit
        assert str(unitaries) == unitaries_from_qiskit
        assert str(orig_ham) == str(mol.hamiltonian)
