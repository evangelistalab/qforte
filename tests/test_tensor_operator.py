from qforte import *
from fqe import fqe_decorators
from openfermion import FermionOperator, hermitian_conjugated
import qforte as qf
import numpy as np
import unittest
import random
import os

class TestTensor(unittest.TestCase):

    @unittest.skip
    def test_tensor_operator(self):        
        
        my_op = qf.SQOperator()
        my_op2 = qf.SQOperator()

        test_tensor1 = qf.Tensor([5, 5], "Test Tensor 1")
        test_tensor2 = qf.Tensor([5, 5], "Test Tensor 2")

        random.seed(123)

        for i in range(5):
            for j in range(5):
                # random.seed(i + j)
                rand = random.uniform(0.1, 10.0)
                my_op.add_term(rand, [i], [j])
                my_op.add_term(rand, [j], [i])


        # my_op.simplify()

        a = np.load('zip_files/norm_test0.npz')

        ablk1, bblk1 = my_op.get_largest_alfa_beta_indices()
        dim1 = (max(ablk1, bblk1) + 1)
        lst1 = my_op.split_by_rank(True)

        t_op = qf.TensorOperator(1, dim1)
        t_op2 = qf.TensorOperator(1, dim1)
        
        t_op.add_sqop_of_rank(lst1[0], 2)

        data = a['data0']

        t_op2.fill_tensor_from_np_by_rank(1, data.ravel(), [5, 5])

        test_tensor1 = t_op.tensors()[1]

        print(t_op2.tensors()[1])

        test_tensor2 = t_op2.tensors()[1]

        test_tensor1.subtract(test_tensor2)

        diff_norm = test_tensor1.norm()

        self.assertLess(diff_norm, 1e-16)

    @unittest.skip
    def test_tensor_2(self):


        my_sqop = qf.SQOperator()
        shape = [5, 5]

        my_tensor = qf.Tensor(shape, "My Tensor")
        imported_tensor = qf.Tensor(shape, "Imported Tensor")

        random.seed(999)

        for i in range(shape[0]):
            for j in range(shape[1]):
                rand_coeff = random.uniform(0.1, 10.0)
                my_sqop.add_term(rand_coeff, [i], [j])
                my_sqop.add_term(rand_coeff, [j], [i])

        a = np.load('zip_files/norm_test5.npz')
        data = a['data0']

        ablk1, bblk1 = my_sqop.get_largest_alfa_beta_indices()
        dim = (max(ablk1, bblk1) + 1)
        split_list = my_sqop.split_by_rank(True)

        my_operator = qf.TensorOperator(1, dim)
        imported_operator = qf.TensorOperator(1, dim)


        imported_operator.fill_tensor_from_np_by_rank(1, data.ravel(), shape)
        my_operator.add_sqop_of_rank(split_list[0], 2)

        my_ref_tensor = my_operator.tensors()[1]
        imported_ref_tensor = imported_operator.tensors()[1]


        my_ref_tensor.subtract(imported_ref_tensor)
        diff_norm = my_ref_tensor.norm()
        
        self.assertLess(diff_norm, 1e-16)
        
    def test_tensor_3(self):
        my_sqop = qf.SQOperator()
        shape = [8, 8, 8, 8]

        my_tensor = qf.Tensor(shape, "My Tensor")
        imported_tensor = qf.Tensor(shape, "Imported Tensor")

        random.seed(999)

        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    for l in range(shape[3]):
                        rand_coeff = random.uniform(0.1, 10.0)
                        my_sqop.add_term(rand_coeff, [i, j], [k, l])
                        my_sqop.add_term(rand_coeff, [l, k], [j, i])

        a = np.load('zip_files/norm_test8.npz')
        data = a['data0']

        ablk1, bblk1 = my_sqop.get_largest_alfa_beta_indices()
        dim = (max(ablk1, bblk1) + 1)
        split_list = my_sqop.split_by_rank(True)

        my_operator = qf.TensorOperator(2, dim)
        imported_operator = qf.TensorOperator(2, dim)

        imported_operator.fill_tensor_from_np_by_rank(4, data.ravel(), shape)

        my_operator.add_sqop_of_rank(split_list[0], 4)

        my_ref_tensor = my_operator.tensors()[1]
        imported_ref_tensor = imported_operator.tensors()[1]

        my_ref_tensor.subtract(imported_ref_tensor)
        diff_norm = my_ref_tensor.norm()
        
        self.assertLess(diff_norm, 1e-16)        

        



unittest.main()

