from qforte import SQOperator, Computer, Tensor
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

        shape = [2, 2]


        t1 = qf.Tensor(shape, "Tensor 1")
        t2 = qf.Tensor(shape, "Tensor 2")

        t1.set([0, 0], 1.0)
        t1.set([1, 1], 1.0)

        t2.set([0, 0], 2.0)
        t2.set([1, 1], 2.0)

        t1.zaxpby(t2, 1.0, 5.0)

    def test_gemm1(self):

        shape = [5, 5]

        t1 = qf.Tensor(shape, "Tensor 1")
        t2 = qf.Tensor(shape, "Tensor 2")

        random_arr1 = np.random.rand(shape[0], shape[1])
        random_arr2 = np.random.rand(shape[0], shape[1])

        for i in range(shape[0]):
            for j in range(shape[1]):
                t1.set([i, j], random_arr1[i, j])
                t2.set([i, j], random_arr2[i, j])

        
        final_np_arr = np.matmul(random_arr1, random_arr2)

        t1.gemm(t2)

        ref_arr = [ [0]*shape[1] for i in range(shape[0])]

        for i in range(shape[0]):
            for j in range(shape[1]):
                ref_arr[i][j] = t1.get([i, j])

        final_norm = np.linalg.norm(final_np_arr) - np.linalg.norm(ref_arr)

        self.assertLess(final_norm, 1e-16)

    def test_gemm2(self):

        shape = [10, 10]

        t1 = qf.Tensor(shape, "Tensor 1")
        t2 = qf.Tensor(shape, "Tensor 2")

        random_arr1 = np.random.rand(shape[0], shape[1])
        random_arr2 = np.random.rand(shape[0], shape[1])

        for i in range(shape[0]):
            for j in range(shape[1]):
                t1.set([i, j], random_arr1[i, j])
                t2.set([i, j], random_arr2[i, j])

        
        final_np_arr = np.matmul(random_arr1, random_arr2)

        t1.gemm(t2)

        ref_arr = [ [0]*shape[1] for i in range(shape[0])]

        for i in range(shape[0]):
            for j in range(shape[1]):
                ref_arr[i][j] = t1.get([i, j])

        final_norm = np.linalg.norm(final_np_arr) - np.linalg.norm(ref_arr)

        self.assertLess(final_norm, 1e-16)

    def test_gemm3(self):

        shape = [10, 10]

        t1 = qf.Tensor(shape, "Tensor 1")
        t2 = qf.Tensor(shape, "Tensor 2")

        random_arr1 = np.random.rand(shape[0], shape[1])
        random_arr2 = np.random.rand(shape[0], shape[1])

        for i in range(shape[0]):
            for j in range(shape[1]):
                t1.set([i, j], random_arr1[i, j])
                t2.set([i, j], random_arr2[i, j])

        
        final_np_arr = np.matmul(random_arr2, random_arr1)

        t1.gemm(t2, mult_B_on_right = True)

        ref_arr = [ [0]*shape[1] for i in range(shape[0])]

        for i in range(shape[0]):
            for j in range(shape[1]):
                ref_arr[i][j] = t1.get([i, j])

        final_norm = np.linalg.norm(final_np_arr) - np.linalg.norm(ref_arr)

        self.assertLess(final_norm, 1e-16)    

    def test_gemm4(self):

        shape = [10, 10]

        t1 = qf.Tensor(shape, "Tensor 1")
        t2 = qf.Tensor(shape, "Tensor 2")

        random_arr1 = np.random.rand(shape[0], shape[1])
        random_arr2 = np.random.rand(shape[0], shape[1])

        random_arr1 = np.array(random_arr1, dtype=complex)
        random_arr2 = np.array(random_arr2, dtype=complex)

        random_arr1[2, 3] = 1.0+2.3j
        random_arr2[1, 0] = 2.5+3.0j

        for i in range(shape[0]):
            for j in range(shape[1]):
                t1.set([i, j], random_arr1[i, j])
                t2.set([i, j], random_arr2[i, j])

        
        final_np_arr = np.matmul(random_arr1.conj().T, random_arr2.conj().T)

        t1.gemm(t2, transa = 'C', transb = 'C')

        ref_arr = [ [0]*shape[1] for i in range(shape[0])]

        for i in range(shape[0]):
            for j in range(shape[1]):
                ref_arr[i][j] = t1.get([i, j])

        final_norm = np.linalg.norm(final_np_arr) - np.linalg.norm(ref_arr)

        self.assertLess(final_norm, 1e-16)  


unittest.main()