from qforte import *
from fqe import fqe_decorators
from openfermion import FermionOperator, hermitian_conjugated
import qforte as qf
import numpy as np
import unittest
import os

class TestTensor(unittest.TestCase):

    def test_tensor_operator(self):        

        my_op = qf.SQOperator()
        my_op.add_term(1.0, [2, 3, 5], [1, 2, 0])
        my_op.add_term(1.0, [0, 2, 1], [5, 3, 2])
        my_op.add_term(1.5, [3, 4], [2, 5])
        my_op.add_term(1.5, [5, 2], [4, 3])

        a = np.load('zip_files/test_zip0.npz')
        b = np.load('zip_files/test_zip1.npz')
        ablk, bblk = my_op.get_largest_alfa_beta_indices()

        dim = max(ablk, bblk) + 1

        list = my_op.split_by_rank(False)

        t_op = qf.TensorOperator(3, dim)
        t_op.add_sqop_of_rank(list[0], 4)
        t_op.add_sqop_of_rank(list[1], 6)

        # print(a['data0'])

        np.testing.assert_array_equal(t_op.tensors()[0], a['data0'])



        



unittest.main()

