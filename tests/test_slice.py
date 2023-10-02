from qforte import *
import qforte as qf
import numpy as np
import unittest



class TestSlice(unittest.TestCase):

    def test1(self):

        shape = [5, 5]

        t1 = qf.Tensor(shape, "Tensor 1")

        for i in range(shape[0]):
            for j in range(shape[1]):
                t1.set([i, j], i + j)

        t2 = t1.slice([(2, 3), (2, 3)])

    def test2(self):

        shape = [6, 7, 8]

        t1 = qf.Tensor(shape, "Tensor 1")

        random_arr = np.random.rand(6, 7, 8)

        for i in range(6):
            for j in range(7):
                for k in range(8):
                    t1.set([i, j, k], random_arr[i, j, k])

        t2 = t1.slice([(2, 3), (2, 4), (2, 3)])
        random_arr_sliced = random_arr[2:3, 2:4, 2:3]

        ref_arr = np.zeros((6, 7, 8), dtype = complex)

        for i in range(t2.shape()[0]):
            for j in range(t2.shape()[1]):
                for k in range(t2.shape()[2]):
                    ref_arr[i, j, k] = t2.get([i, j, k])

        final_norm = np.linalg.norm(random_arr_sliced) - np.linalg.norm(ref_arr)

        self.assertLess(final_norm, 1e-16)

    def test3(self):

        shape = [12, 13, 14, 18]

        t1 = qf.Tensor(shape, "Tensor 1")

        random_arr = np.random.rand(12, 13, 14, 18)

        for i in range(12):
            for j in range(13):
                for k in range(14):
                        for l in range(18):
                            t1.set([i, j, k, l], random_arr[i, j, k, l])

        t2 = t1.slice([(2, 8), (5, 10), (10, 11), (5, 6)])
        random_arr_sliced = random_arr[2:8, 5:10, 10:11, 5:6]

        ref_arr = np.zeros((12, 13, 14, 18), dtype = complex)

        for i in range(t2.shape()[0]):
            for j in range(t2.shape()[1]):
                for k in range(t2.shape()[2]):
                    for l in range(t2.shape()[3]):
                        ref_arr[i, j, k, l] = t2.get([i, j, k, l])

        final_norm = np.linalg.norm(random_arr_sliced) - np.linalg.norm(ref_arr)

        self.assertLess(final_norm, 1e-16)

    def test4(self):

        shape = [12, 13, 14, 18]

        t1 = qf.Tensor(shape, "Tensor 1")

        random_arr = np.random.rand(12, 13, 14, 18)

        for i in range(12):
            for j in range(13):
                for k in range(14):
                        for l in range(18):
                            t1.set([i, j, k, l], random_arr[i, j, k, l])

        t2 = t1.slice([(0, 12), (0, 13), (0, 14), (0, 18)])

        ref_arr = np.zeros((12, 13, 14, 18), dtype = complex)
        ref_arr2 = np.zeros((12, 13, 14, 18), dtype = complex)

        for i in range(t2.shape()[0]):
            for j in range(t2.shape()[1]):
                for k in range(t2.shape()[2]):
                    for l in range(t2.shape()[3]):
                        ref_arr[i, j, k, l] = t2.get([i, j, k, l])

        for i in range(t1.shape()[0]):
            for j in range(t1.shape()[1]):
                for k in range(t1.shape()[2]):
                    for l in range(t1.shape()[3]):
                        ref_arr2[i, j, k, l] = t1.get([i, j, k, l])

        final_norm = np.linalg.norm(ref_arr2) - np.linalg.norm(ref_arr)

        self.assertLess(final_norm, 1e-15)

    def test5(self):

        shape = [12, 13, 14, 18]

        t1 = qf.Tensor(shape, "Tensor 1")

        random_arr = np.random.rand(12, 13, 14, 18)

        for i in range(12):
            for j in range(13):
                for k in range(14):
                        for l in range(18):
                            t1.set([i, j, k, l], random_arr[i, j, k, l])

        t2 = t1.slice([(6, 12), (7, 13), (7, 14), (9, 18)])
        random_arr_sliced = random_arr[6:12, 7:13, 7:14, 9:18]

        ref_arr = np.zeros((12, 13, 14, 18), dtype = complex)

        for i in range(t2.shape()[0]):
            for j in range(t2.shape()[1]):
                for k in range(t2.shape()[2]):
                    for l in range(t2.shape()[3]):
                        ref_arr[i, j, k, l] = t2.get([i, j, k, l])

        final_norm = np.linalg.norm(random_arr_sliced) - np.linalg.norm(ref_arr)

        self.assertLess(final_norm, 1e-14)

    def test6(self):

        shape = [12, 13, 14, 18]

        t1 = qf.Tensor(shape, "Tensor 1")

        random_arr = np.random.rand(12, 13, 14, 18)


        for i in range(12):
            for j in range(13):
                for k in range(14):
                        for l in range(18):
                            t1.set([i, j, k, l], random_arr[i, j, k, l])

        t2 = t1.slice([(0, 5), (0, 6), (0, 6), (0, 8)])
        random_arr_sliced = random_arr[0:5, 0:6, 0:6, 0:8]

        ref_arr = np.zeros((12, 13, 14, 18), dtype = complex)

        for i in range(t2.shape()[0]):
            for j in range(t2.shape()[1]):
                for k in range(t2.shape()[2]):
                    for l in range(t2.shape()[3]):
                        ref_arr[i, j, k, l] = t2.get([i, j, k, l])

        final_norm = np.linalg.norm(random_arr_sliced) - np.linalg.norm(ref_arr)

        self.assertLess(final_norm, 1e-14)

    def test7(self):

        shape = [12, 13, 14, 18]

        t1 = qf.Tensor(shape, "Tensor 1")

        random_arr = np.random.rand(12, 13, 14, 18)
        random_arr = np.array(random_arr, dtype = complex)
        

        for i in range(12):
            for j in range(13):
                for k in range(14):
                        for l in range(18):
                            t1.set([i, j, k, l], random_arr[i, j, k, l])

        t1.set([2, 3, 4, 2], 1.5+2.0j)
        random_arr[2, 3, 4, 2] = 1.5+2.0j

        t2 = t1.slice([(0, 5), (0, 6), (0, 6), (0, 8)])
        random_arr_sliced = random_arr[0:5, 0:6, 0:6, 0:8]

        ref_arr = np.zeros((12, 13, 14, 18), dtype = complex)

        for i in range(t2.shape()[0]):
            for j in range(t2.shape()[1]):
                for k in range(t2.shape()[2]):
                    for l in range(t2.shape()[3]):
                        ref_arr[i, j, k, l] = t2.get([i, j, k, l])

        final_norm = np.linalg.norm(random_arr_sliced) - np.linalg.norm(ref_arr)

        self.assertLess(final_norm, 1e-14)
