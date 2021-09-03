import unittest
import qforte as qf
import numpy as np

class SparseOpTests(unittest.TestCase):
    def test_sparse_operator(self):

        print('\nSTART test_sparse_operator\n')

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
        -0.062500j
        ]

        circ_vec = [
        qf.build_circuit('X_1 Y_2 X_3 Y_4'),
        qf.build_circuit('X_1 Y_2 X_3 X_4'),
        qf.build_circuit('X_1 Y_2 Y_3 X_4'),
        qf.build_circuit('X_1 X_2 X_3 Y_4'),
        qf.build_circuit('X_1 X_2 X_3 X_4'),
        qf.build_circuit('X_1 X_2 Y_3 X_4'),
        qf.build_circuit('Y_2 X_3 X_4 Z_5 X_6'),
        qf.build_circuit('Y_1 Y_2 Y_3 Y_4'),
        qf.build_circuit('Y_1 X_2 X_3 Y_4'),
        qf.build_circuit('Y_1 X_2 X_3 X_4'),
        qf.build_circuit('Y_1 Y_2 X_3 X_4'),
        qf.build_circuit('X_2 X_3 X_4 Z_5 X_6'),
        qf.build_circuit('Y_1 X_2 Y_3 Y_4'),
        qf.build_circuit('X_2 Y_3 Y_4 Z_5 Y_6'),
        qf.build_circuit('X_2 Y_3 X_4 Z_5 X_6'),
        qf.build_circuit('X_2 Y_3 Y_4 Z_5 X_6'),
        qf.build_circuit('X_2 X_3 Y_4 Z_5 Y_6'),
        qf.build_circuit('Y_2 Y_3 X_4 Z_5 X_6'),
        qf.build_circuit('X_2 X_3 Y_4 Z_5 X_6'),
        qf.build_circuit('Y_1 X_2 Y_3 X_4'),
        qf.build_circuit('Y_2 Y_3 X_4 Z_5 Y_6'),
        qf.build_circuit('Y_2 X_3 X_4 Z_5 Y_6'),
        qf.build_circuit('X_2 X_3 X_4 Z_5 Y_6'),
        qf.build_circuit('X_2 Y_3 X_4 Z_5 Y_6'),
        qf.build_circuit('Y_1 Y_2 Y_3 X_4'),
        qf.build_circuit('Y_1 Y_2 X_3 Y_4'),
        qf.build_circuit('Y_2 Y_3 Y_4 Z_5 X_6'),
        qf.build_circuit('X_1 X_2 Y_3 Y_4'),
        qf.build_circuit('Y_2 Y_3 Y_4 Z_5 Y_6'),
        qf.build_circuit('Y_2 X_3 Y_4 Z_5 X_6'),
        qf.build_circuit('X_1 Y_2 Y_3 Y_4'),
        qf.build_circuit('Y_2 X_3 Y_4 Z_5 Y_6')
        ]

        qubit_op = qf.QubitOperator()
        for coeff, circ in zip(coeff_vec, circ_vec):
            qubit_op.add(coeff, circ)

        num_qb = qubit_op.num_qubits()
        qci = qf.Computer(num_qb)
        arb_vec = np.linspace(0, 2*np.pi, 2**num_qb)
        arb_vec = arb_vec/np.linalg.norm(arb_vec)
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
        print('Operator used for sparse matrix operator test: \n', qubit_op)
        print('||∆||:          ', diff_val)
        self.assertTrue(diff_val < 1.0e-15)

if __name__ == '__main__':
    unittest.main()
