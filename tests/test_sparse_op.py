from qforte import build_circuit, Computer, QubitOperator
import numpy as np
import qforte as qf
import random
from copy import deepcopy


class TestSparseOp:
    def test_sparse_operator(self):
        """
        test the SparseMatrix and SparseVector classes
        """

        coeff_vec = [
            -0.093750,
            +0.093750j,
            -0.093750,
            -0.093750j,
            -0.093750,
            -0.093750j,
            +0.062500j,
            -0.093750,
            -0.093750,
            +0.093750j,
            +0.093750,
            -0.062500,
            -0.093750j,
            -0.062500j,
            +0.062500j,
            -0.062500,
            +0.062500,
            +0.062500,
            -0.062500j,
            -0.093750,
            +0.062500j,
            -0.062500,
            -0.062500j,
            -0.062500,
            +0.093750j,
            +0.093750j,
            +0.062500j,
            +0.093750,
            -0.062500,
            -0.062500,
            -0.093750j,
            -0.062500j,
        ]

        circ_vec = [
            build_circuit("X_1 Y_2 X_3 Y_4"),
            build_circuit("X_1 Y_2 X_3 X_4"),
            build_circuit("X_1 Y_2 Y_3 X_4"),
            build_circuit("X_1 X_2 X_3 Y_4"),
            build_circuit("X_1 X_2 X_3 X_4"),
            build_circuit("X_1 X_2 Y_3 X_4"),
            build_circuit("Y_2 X_3 X_4 Z_5 X_6"),
            build_circuit("Y_1 Y_2 Y_3 Y_4"),
            build_circuit("Y_1 X_2 X_3 Y_4"),
            build_circuit("Y_1 X_2 X_3 X_4"),
            build_circuit("Y_1 Y_2 X_3 X_4"),
            build_circuit("X_2 X_3 X_4 Z_5 X_6"),
            build_circuit("Y_1 X_2 Y_3 Y_4"),
            build_circuit("X_2 Y_3 Y_4 Z_5 Y_6"),
            build_circuit("X_2 Y_3 X_4 Z_5 X_6"),
            build_circuit("X_2 Y_3 Y_4 Z_5 X_6"),
            build_circuit("X_2 X_3 Y_4 Z_5 Y_6"),
            build_circuit("Y_2 Y_3 X_4 Z_5 X_6"),
            build_circuit("X_2 X_3 Y_4 Z_5 X_6"),
            build_circuit("Y_1 X_2 Y_3 X_4"),
            build_circuit("Y_2 Y_3 X_4 Z_5 Y_6"),
            build_circuit("Y_2 X_3 X_4 Z_5 Y_6"),
            build_circuit("X_2 X_3 X_4 Z_5 Y_6"),
            build_circuit("X_2 Y_3 X_4 Z_5 Y_6"),
            build_circuit("Y_1 Y_2 Y_3 X_4"),
            build_circuit("Y_1 Y_2 X_3 Y_4"),
            build_circuit("Y_2 Y_3 Y_4 Z_5 X_6"),
            build_circuit("X_1 X_2 Y_3 Y_4"),
            build_circuit("Y_2 Y_3 Y_4 Z_5 Y_6"),
            build_circuit("Y_2 X_3 Y_4 Z_5 X_6"),
            build_circuit("X_1 Y_2 Y_3 Y_4"),
            build_circuit("Y_2 X_3 Y_4 Z_5 Y_6"),
        ]

        qubit_op = QubitOperator()
        for coeff, circ in zip(coeff_vec, circ_vec):
            qubit_op.add(coeff, circ)

        num_qb = qubit_op.num_qubits()
        qci = Computer(num_qb)
        arb_vec = np.linspace(0, 2 * np.pi, 2**num_qb)
        arb_vec = arb_vec / np.linalg.norm(arb_vec)
        qci.set_coeff_vec(arb_vec)
        sp_mat_op = qubit_op.sparse_matrix(num_qb)
        ci = np.array(qci.get_coeff_vec())
        qci.apply_operator(qubit_op)
        diff_vec = np.array(qci.get_coeff_vec())

        # Reset stae
        qci.set_coeff_vec(ci)
        qci.apply_sparse_matrix(sp_mat_op)

        # see if there is a difference
        diff_vec = np.array(qci.get_coeff_vec()) - diff_vec
        diff_val = np.linalg.norm(diff_vec)
        print("Operator used for sparse matrix operator test: \n", qubit_op)
        print("||∆||:          ", diff_val)
        assert diff_val < 1.0e-15

    def test_sparse_gates(self):
        """
        This test ensures that the sparse matrix representations of various
        gates agrees with those obtained using the kronecker product with
        the identity matrix
        """

        def sparse_matrix_to_numpy_array(sp_mat: dict, dim: int) -> np.array:
            mat = np.zeros((dim, dim), dtype=complex)
            for row in sp_mat:
                for column in sp_mat[row]:
                    mat[row, column] = sp_mat[row][column]
            return mat

        one_qubit_gate_pool = [
            "X",
            "Y",
            "Z",
            "Rx",
            "Ry",
            "Rz",
            "H",
            "S",
            "T",
            "R",
            "V",
            "adj(V)",
        ]

        two_qubit_gate_pool = [
            "CNOT",
            "aCNOT",
            "cY",
            "cZ",
            "cV",
            "SWAP",
            "cRz",
            "cR",
            "A",
            "adj(cV)",
        ]

        parametrized_gate_periods = {
            "Rx": 4 * np.pi,
            "Ry": 4 * np.pi,
            "Rz": 4 * np.pi,
            "R": 2 * np.pi,
            "cR": 2 * np.pi,
            "cRz": 4 * np.pi,
            "A": 2 * np.pi,
        }

        dim = 2
        identity = np.eye(2)
        for gatetype in one_qubit_gate_pool:
            if gatetype in parametrized_gate_periods:
                period = parametrized_gate_periods[gatetype]
                parameter = random.uniform(-period / 2, period / 2)
                gate_0 = qf.gate(gatetype, 0, parameter)
                gate_1 = qf.gate(gatetype, 1, parameter)
            else:
                if gatetype == "adj(V)":
                    gate_0 = qf.gate("V", 0)
                    gate_0 = gate_0.adjoint()
                    gate_1 = qf.gate("V", 1)
                    gate_1 = gate_1.adjoint()
                else:
                    gate_0 = qf.gate(gatetype, 0)
                    gate_1 = qf.gate(gatetype, 1)
            sparse_mat_0 = gate_0.sparse_matrix(2)
            sparse_mat_1 = gate_1.sparse_matrix(2)
            mat = sparse_matrix_to_numpy_array(gate_0.sparse_matrix(1).to_map(), dim)
            mat_0 = sparse_matrix_to_numpy_array(sparse_mat_0.to_map(), dim * 2)
            mat_1 = sparse_matrix_to_numpy_array(sparse_mat_1.to_map(), dim * 2)
            mat_0_kron = np.kron(identity, mat)
            mat_1_kron = np.kron(mat, identity)
            assert np.all(mat_0 == mat_0_kron)
            assert np.all(mat_1 == mat_1_kron)

        dim = 4
        for gatetype in two_qubit_gate_pool:
            if gatetype in parametrized_gate_periods:
                period = parametrized_gate_periods[gatetype]
                parameter = random.uniform(-period / 2, period / 2)
                gate_01 = qf.gate(gatetype, 0, 1, parameter)
                gate_10 = qf.gate(gatetype, 1, 0, parameter)
                gate_12 = qf.gate(gatetype, 1, 2, parameter)
                gate_21 = qf.gate(gatetype, 2, 1, parameter)
            else:
                if gatetype == "adj(cV)":
                    gate_01 = qf.gate("cV", 0, 1)
                    gate_01 = gate_01.adjoint()
                    gate_10 = qf.gate("cV", 1, 0)
                    gate_10 = gate_10.adjoint()
                    gate_12 = qf.gate("cV", 1, 2)
                    gate_12 = gate_12.adjoint()
                    gate_21 = qf.gate("cV", 2, 1)
                    gate_21 = gate_21.adjoint()
                else:
                    gate_01 = qf.gate(gatetype, 0, 1)
                    gate_10 = qf.gate(gatetype, 1, 0)
                    gate_12 = qf.gate(gatetype, 1, 2)
                    gate_21 = qf.gate(gatetype, 2, 1)
            sparse_mat_01 = gate_01.sparse_matrix(3)
            sparse_mat_10 = gate_10.sparse_matrix(3)
            sparse_mat_12 = gate_12.sparse_matrix(3)
            sparse_mat_21 = gate_21.sparse_matrix(3)
            mat1 = sparse_matrix_to_numpy_array(gate_01.sparse_matrix(2).to_map(), dim)
            mat2 = sparse_matrix_to_numpy_array(gate_10.sparse_matrix(2).to_map(), dim)
            mat_01 = sparse_matrix_to_numpy_array(sparse_mat_01.to_map(), dim * 2)
            mat_10 = sparse_matrix_to_numpy_array(sparse_mat_10.to_map(), dim * 2)
            mat_12 = sparse_matrix_to_numpy_array(sparse_mat_12.to_map(), dim * 2)
            mat_21 = sparse_matrix_to_numpy_array(sparse_mat_21.to_map(), dim * 2)
            mat_01_kron = np.kron(identity, mat1)
            mat_10_kron = np.kron(identity, mat2)
            mat_12_kron = np.kron(mat1, identity)
            mat_21_kron = np.kron(mat2, identity)
            assert np.all(mat_01 == mat_01_kron)
            assert np.all(mat_10 == mat_10_kron)
            assert np.all(mat_12 == mat_12_kron)
            assert np.all(mat_21 == mat_21_kron)

    def test_circuit_sparse_matrix(self):
        Rhh = 1.5

        mol = qf.system_factory(
            system_type="molecule",
            build_type="psi4",
            basis="sto-6g",
            mol_geometry=[
                ("H", (0, 0, -3 * Rhh / 2)),
                ("H", (0, 0, -Rhh / 2)),
                ("H", (0, 0, Rhh / 2)),
                ("H", (0, 0, 3 * Rhh / 2)),
            ],
            symmetry="d2h",
            multiplicity=1,
            charge=0,
            num_frozen_docc=0,
            num_frozen_uocc=0,
            run_mp2=0,
            run_ccsd=0,
            run_cisd=0,
            run_fci=0,
        )

        for compact in [False, True]:
            uccsd = qf.UCCNVQE(
                mol, compact_excitations=compact, qubit_excitations=False
            )
            uccsd.run(
                pool_type="SD", opt_maxiter=200, optimizer="bfgs", opt_thresh=1.0e-5
            )

            qc1 = qf.Computer(mol.hamiltonian.num_qubits())
            qc1.apply_circuit(uccsd.build_Uvqc())
            coeff_vec1 = deepcopy(qc1.get_coeff_vec())

            Uvqc = uccsd.build_Uvqc()
            sparse_Uvqc = Uvqc.sparse_matrix(mol.hamiltonian.num_qubits())
            qc2 = qf.Computer(mol.hamiltonian.num_qubits())
            qc2.apply_sparse_matrix(sparse_Uvqc)
            coeff_vec2 = qc2.get_coeff_vec()

            assert np.linalg.norm(np.array(coeff_vec2) - np.array(coeff_vec1)) < 1.0e-14
