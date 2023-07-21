from qforte import *
from fqe import fqe_decorators
from openfermion import FermionOperator, hermitian_conjugated
import qforte as qf
import numpy as np
import unittest
import random
import os

class TestTensor(unittest.TestCase):

    def test_tensor_operator(self):        
        
        my_op = qf.SQOperator()
        my_op2 = qf.SQOperator()
        random.seed(123)

        for i in range(5):
            for j in range(5):
                # random.seed(i + j)
                rand = random.uniform(0.1, 10.0)
                my_op.add_term(rand, [i], [j])
                my_op.add_term(rand, [j], [i])

        a = np.load('zip_files/norm_test0.npz')
        # b = np.load('zip_files/test_zip1.npz')

        ablk1, bblk1 = my_op.get_largest_alfa_beta_indices()
        dim1 = max(ablk1, bblk1) + 1
        lst1 = my_op.split_by_rank(False)

        t_op = qf.TensorOperator(1, dim1)
        t_op2 = qf.TensorOperator(1, dim1)
        
        t_op.add_sqop_of_rank(lst1[0], 2)

        data = a['data0']

        # print(t_op.tensors()[1].norm())
        
        # print(data.ravel())

        print(t_op2.tensors()[1])

        # filling t_op2 with fill_from_nparray does not work for some reason.
        # after running the line below, the norm is still zero
        # Printing out the contents of data.ravel() shows that data contains numbers
        # But for some reason fill_from_nparray just doesn't work on it

        t_op2.tensors()[1].fill_from_nparray(data.ravel(), [5, 5])

        t_op.tensors()[1].subtract(t_op2.tensors()[1])

        diff_norm = t_op.tensors()[1].norm()

        self.assertLess(diff_norm, 1e-16)


        



unittest.main()

