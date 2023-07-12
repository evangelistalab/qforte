from qforte import SQOperator, Computer
import qforte as qf
import numpy as np
import unittest

class TestApplySQOP(unittest.TestCase):


    def test_single_creator1(self):

        # Initialize reference vector
        qc1 = qf.Computer(4)
        vec1 = qc1.get_coeff_vec()
        vec1[4] = 1.0
        vec1[0] = 0

        # Initialize Second Quantized Operator
        my_op = SQOperator()
        my_op.add_term(1.0, [2], [])

        # Apply SQOperator to second computer
        qc2 = qf.Computer(4)
        qc2.apply_sq_operator(my_op)

        # Take the difference of the norms
        diff_vec = np.asarray(vec1) - np.asarray(qc2.get_coeff_vec())

        # Check if it is less than Machine Epsilon
        self.assertLess(np.linalg.norm(diff_vec), 1e-16)

    def test_single_creator2(self):

        # Initialize reference vector
        qc1 = qf.Computer(6)
        vec1 = qc1.get_coeff_vec()
        vec1[32] = 1.0
        vec1[0] = 0

        # Initialize Second Quantized Operator
        my_op = SQOperator()
        my_op.add_term(1.0, [5], [])

        # Apply SQOperator to second computer
        qc2 = qf.Computer(6)
        qc2.apply_sq_operator(my_op)

        # Take the difference of the norms
        diff_vec = np.asarray(vec1) - np.asarray(qc2.get_coeff_vec())

        # Check if it is less than Machine Epsilon
        self.assertLess(np.linalg.norm(diff_vec), 1e-16)

    def test_single_annihilator1(self):
        
        qc1 = qf.Computer(6)
        vec1 = qc1.get_coeff_vec()
        vec1[0] = 0
        vec1[8] = 1.0

        my_op = qf.SQOperator()
        my_op.add_term(1.0, [], [3])

        qc2 = qf.Computer(6)
        qc2.set_coeff_vec(vec1)
        qc2.apply_sq_operator(my_op)

        self.assertEqual(np.linalg.norm(np.asarray(qc2.get_coeff_vec())), 1.0)

    def test_single_annihilator2(self):

        qc1 = qf.Computer(6)
        vec1 = qc1.get_coeff_vec()
        vec1[0] = 0
        vec1[16] = 1.0

        my_op = qf.SQOperator()
        my_op.add_term(1.0, [], [4])

        qc2 = qf.Computer(6)
        qc2.set_coeff_vec(vec1)
        qc2.apply_sq_operator(my_op)

        self.assertEqual(np.linalg.norm(np.asarray(qc2.get_coeff_vec())), 1.0)

    def test_excitation1(self):

        qc1 = qf.Computer(6)
        vec1 = qc1.get_coeff_vec()
        vec2 = qc1.get_coeff_vec()

        vec1[0] = 0
        vec1[3] = 1.0

        vec2[0] = 0
        vec2[12] = 1.0


        my_op = qf.SQOperator()
        my_op.add_term(1.0, [2, 3], [1, 0])

        qc2 = qf.Computer(6)
        qc2.set_coeff_vec(vec1)
        qc2.apply_sq_operator(my_op)

        diff_vec = np.asarray(vec2) - np.asarray(qc2.get_coeff_vec())

        self.assertLess(np.linalg.norm(diff_vec), 1e-16)

    def test_excitation2(self):

        qc1 = qf.Computer(6)
        vec1 = qc1.get_coeff_vec()
        vec2 = qc1.get_coeff_vec()

        # 1 0 1 0 0          0 0 1 0 1 # 

        vec1[0] = 0
        vec1[5] = 1.0

        vec2[0] = 0
        vec2[20] = 1.0

        my_op = qf.SQOperator()
        my_op.add_term(1.0, [2, 4], [2, 0])

        qc2 = qf.Computer(6)
        qc2.set_coeff_vec(vec1)
        qc2.apply_sq_operator(my_op)

        diff_vec = np.asarray(vec2) - np.asarray(qc2.get_coeff_vec())

        self.assertLess(np.linalg.norm(diff_vec), 1e-16)


    def test_wipe1(self):

        qc1 = qf.Computer(6)
        vec1 = qc1.get_coeff_vec()

        random_state = np.random.rand(6)
        random_state = random_state / np.linalg.norm(random_state)

        vec1[0] = 0

        my_op = qf.SQOperator()
        my_op.add_term(1.0, [2, 2], [])

        qc2 = qf.Computer(6)
        qc2.set_coeff_vec(random_state)

        qc2.apply_sq_operator(my_op)
        
        self.assertEqual(np.linalg.norm(np.asarray(qc2.get_coeff_vec())), 0)


    def test_random1(self):

        qc1 = qf.Computer(6)
        qc2 = qf.Computer(6)

        random_state = np.random.rand(2**6)
        random_state = random_state / np.linalg.norm(random_state)

        qc1.set_coeff_vec(random_state)
        qc2.set_coeff_vec(random_state)

        my_op = qf.SQOperator()
        my_op2 = qf.SQOperator()
        my_op.add_term(1.0, [2, 3], [1, 0])
        my_op2.add_term(1.0, [2, 3], [1, 0])
        pauli_op = my_op2.jw_transform()

        qc1.apply_sq_operator(my_op)
        qc2.apply_operator(pauli_op)

        

        diff_vector = np.asarray(qc1.get_coeff_vec()) - np.asarray(qc2.get_coeff_vec())

        self.assertLess(np.linalg.norm(diff_vector), 1e-15)

    def test_get_largest_alfa_beta_indices(self):


        my_op = qf.SQOperator()
        my_op.add_term(1.0, [1, 3], [2, 4])

        alfa, beta = my_op.get_largest_alfa_beta_indices()

        self.assertEqual(alfa, 4)
        self.assertEqual(beta, 3)

    def test_get_largest_alfa_beta_indices2(self):


        my_op = qf.SQOperator()
        my_op.add_term(1.0, [2, 4], [0, 2])

        alfa, beta = my_op.get_largest_alfa_beta_indices()

        self.assertEqual(alfa, 4)
        self.assertEqual(beta, -1)

    def test_get_largest_alfa_beta_indices3(self):

        my_op = qf.SQOperator()
        my_op.add_term(1.0, [1, 3], [5, 1])

        alfa, beta = my_op.get_largest_alfa_beta_indices()

        self.assertEqual(alfa, -1)
        self.assertEqual(beta, 5)

    def test_many_body_order(self):

        my_op = qf.SQOperator()
        my_op.add_term(1.0, [2, 3, 4, 5, 1, 0], [2, 4, 2, 1, 0])
        my_op.add_term(1.5, [1, 0], [4, 3])
        my_op.add_term(1.0, [2, 3, 3], [4, 2, 1])
        
        order = my_op.many_body_order()

        self.assertEqual(order, 11)

    def test_many_body_order2(self):

        my_op = qf.SQOperator()
        
        order = my_op.many_body_order()

        self.assertEqual(order, -1)

    def test_many_body_order3(self):

        my_op = qf.SQOperator()

        my_op.add_term(2.0, [3, 2, 3, 0], [2, 1])
        my_op.add_term(1.0, [2, 2, 1, 0], [1, 0])

        order = my_op.many_body_order()

        self.assertEqual(order, 6)

    def test_ranks_present(self):

        my_op = qf.SQOperator()

        my_op.add_term(1.0, [2, 3, 4, 5], [1, 0, 2, 1])
        my_op.add_term(2.0, [1, 5, 6, 2, 2], [1, 1, 2, 4, 5])
        my_op.add_term(1.5, [1, 2], [2, 3])

        ranks = my_op.ranks_present()

        self.assertEqual(ranks, [8, 10, 4])

    def test_ranks_present2(self):

        my_op = qf.SQOperator()

        my_op.add_term(1.5, [2, 1, 3], [4, 5])
        my_op.add_term(1.0, [1], [2])

        ranks = my_op.ranks_present()

        self.assertEqual(ranks, [5, 2])


unittest.main()






