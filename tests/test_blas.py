from qforte import SQOperator, Computer
import qforte as qf
import numpy as np
import unittest

class TestBlas(unittest.TestCase):

    def test_zaxpy1(self):
        shape = [4, 4]

        t1 = qf.Tensor(shape, "Tensor 1")
        t1.set([0,0], 2.0)
        t1.set([1,1], 2.0)
        t1.set([2,0], 2.0)

        t2 = qf.Tensor(shape, "Tensor 2")
        t2.set([0, 2], 1.5)
        t2.set([1, 1], 1.5)
        t2.set([2, 1], 1.0+2.0j)

        rt = qf.Tensor(shape, "Ref Tensor")
        rt.set([0, 0], 2.0)
        rt.set([1, 1], 3.5)
        rt.set([0, 2], 1.5)
        rt.set([2, 0], 2.0)
        rt.set([2, 1], 1.0+2.0j)

        t1.zaxpy(t2, 1.0)

        rt.scale(-1)
        rt.add(t1)

        ref_arr = [ [0]*4 for i in range(4)]

        for i in range(4):
            for j in range(4):
                ref_arr[i][j] = rt.get([i, j])
        
        self.assertLess(np.linalg.norm(ref_arr), 1e-16)

    def test_zaxpy2(self):

        shape = [5, 5]

        random_arr = np.random.rand(5, 5)
        random_arr2 = np.random.rand(5, 5)

        random_arr = np.array(random_arr, dtype=complex)
        random_arr2 = np.array(random_arr2, dtype=complex)

        random_arr[2, 3] = 1.0+2.3j
        random_arr2[1, 0] = 2.5+3.0j

        t1 = qf.Tensor(shape, "Tensor 1")
        t2 = qf.Tensor(shape, "Tensor 2")

        for i in range(5):
            for j in range(5):
                t1.set([i, j], random_arr[i, j])
                t2.set([i, j], random_arr2[i, j])

        t1.zaxpy(t2, 1.0)

        random_arr = random_arr + random_arr2

        ref_arr = [ [0]*5 for i in range(5)]

        for i in range(5):
            for j in range(5):
                ref_arr[i][j] = t1.get([i, j])

        self.assertLess(np.linalg.norm(ref_arr - random_arr), 1e-16)


    def test_zaxpby1(self):

        shape = [5, 5]

        random_arr = np.random.rand(5, 5)
        random_arr2 = np.random.rand(5, 5)

        random_arr = np.array(random_arr, dtype=complex)
        random_arr2 = np.array(random_arr2, dtype=complex)

        random_arr[2, 3] = 1.0+2.3j
        random_arr2[1, 0] = 2.5+3.0j

        t1 = qf.Tensor(shape, "Tensor 1")
        t2 = qf.Tensor(shape, "Tensor 2")

        for i in range(5):
            for j in range(5):
                t1.set([i, j], random_arr[i, j])
                t2.set([i, j], random_arr2[i, j])

        t1.zaxpby(t2, 1.0, -2.0)


        random_arr = (random_arr * -2) + (random_arr2 * 1)

        ref_arr = [ [0]*5 for i in range(5)]

        for i in range(5):
            for j in range(5):
                ref_arr[i][j] = t1.get([i, j])

        self.assertLess(np.linalg.norm(ref_arr - random_arr), 1e-16)


    def test_zaxpby2(self):

        # appears to scale 'a' with t2 and 'b' with t1??

        shape = [2, 2]


        t1 = qf.Tensor(shape, "Tensor 1")
        t2 = qf.Tensor(shape, "Tensor 2")

        t1.set([0, 0], 1.0)
        t1.set([1, 1], 1.0)

        t2.set([0, 0], 2.0)
        t2.set([1, 1], 2.0)

        t1.zaxpby(t2, 1.0, 5.0)

        print(t1)




        







#   2   0   0   0
#   0   2   0   0
#   2   0   0   0
#   0   0   0   0

#   0   0   1.5   0
#   0   0   0   1.5
#   0   j   0   0
#   0   0   0   0



unittest.main()