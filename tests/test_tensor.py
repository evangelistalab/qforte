from qforte import *
import qforte as qf
import numpy as np
import unittest

class TestTensor(unittest.TestCase):

    def test_add(self):

        shape = [3, 3]
        ref_arr = [ [0]*3 for i in range(3)]
        ref_arr1 = [ [0]*3 for i in range(3)]
        

        t1 = qf.Tensor(shape, "Tensor 1")
        t1.set([0, 0], 1.0)
        t1.set([1, 1], 2.0)
        t1.set([2, 2], 3.0)

        t2 = qf.Tensor(shape, "Tensor 2")
        t2.set([0, 0], 1.0)
        t2.set([1, 1], 2.0)
        t2.set([2, 2], 3.0)

        t1.add(t2)

        t3 = qf.Tensor(shape, "Tensor Ref")
        t3.set([0, 0], 2.0)
        t3.set([1, 1], 4.0)
        t3.set([2, 2], 6.0)

        # Converting data to array
        for i in range(t1.shape()[0]):
            for j in range(t1.shape()[1]):
                ref_arr[i][j] = t1.get([i, j])


        for i in range(t3.shape()[0]):
            for j in range(t3.shape()[1]):
                ref_arr1[i][j] = t3.get([i, j])
                

        numpy.testing.assert_array_equal(ref_arr, ref_arr1)

    def test_add2(self):
        
        shape = [4, 4]
        ref_arr = [ [0]*4 for i in range(4)]
        ref_arr1 = [ [0]*4 for i in range(4)]
        

        t1 = qf.Tensor(shape, "Tensor 1")
        t1.set([0, 0], 1.0)
        t1.set([1, 1], 1.0+1.0j)
        t1.set([2, 2], 1.0)
        t1.set([3, 3], 4.0)

        t2 = qf.Tensor(shape, "Tensor 2")
        t2.set([0, 0], 3.5)
        t2.set([1, 1], 3.5)
        t2.set([2, 2], 3.5)
        t2.set([3, 3], 4.5)

        t1.add(t2)

        t3 = qf.Tensor(shape, "Tensor Ref")
        t3.set([0, 0], 4.5)
        t3.set([1, 1], 4.5+1.0j)
        t3.set([2, 2], 4.5)
        t3.set([3, 3], 8.5)

        # Converting data to array
        for i in range(t1.shape()[0]):
            for j in range(t1.shape()[1]):
                ref_arr[i][j] = t1.get([i, j])

        for i in range(t3.shape()[0]):
            for j in range(t3.shape()[1]):
                ref_arr1[i][j] = t3.get([i, j])
                

        numpy.testing.assert_array_equal(ref_arr, ref_arr1)

    def test_add3(self):

        shape = [10, 12]
        ref_arr = [ [0]*12 for i in range(10)]

        t1 = qf.Tensor(shape, "Tensor 1")
        t2 = qf.Tensor(shape, "Tensor 2")

        random_array = np.random.rand(10, 12)
        random_array2 = np.random.rand(10, 12)

        
        for i in range(shape[0]):
            for j in range(shape[1]):
                t1.set([i, j], random_array[i, j])
                t2.set([i, j], random_array2[i, j])

        t1.add(t2)

        for i in range(shape[0]):
            for j in range(shape[1]):
                ref_arr[i][j] = t1.get([i, j])

        final_array = np.add(random_array, random_array2) - ref_arr

        self.assertLess(np.linalg.norm(final_array), 1e-16)
        

    def test_add4(self):
        shape1 = [3, 2]
        t1 = qf.Tensor(shape1, "Tensor 1")

        shape2 = [1, 4]
        t2 = qf.Tensor(shape2, "Tensor 2")

        with self.assertRaises(RuntimeError):
            t1.add(t2)

    def test_subtract(self):
        
        shape = [10, 12]
        ref_arr = [ [0]*12 for i in range(10)]

        t1 = qf.Tensor(shape, "Tensor 1")
        t2 = qf.Tensor(shape, "Tensor 2")

        random_array = np.random.rand(10, 12)
        random_array2 = np.random.rand(10, 12)

        
        for i in range(shape[0]):
            for j in range(shape[1]):
                t1.set([i, j], random_array[i, j])
                t2.set([i, j], random_array2[i, j])

        t1.subtract(t2)

        for i in range(shape[0]):
            for j in range(shape[1]):
                ref_arr[i][j] = t1.get([i, j])

        final_array = np.subtract(random_array, random_array2) - ref_arr

        self.assertLess(np.linalg.norm(final_array), 1e-16)

    def test_subtract2(self):
        shape1 = [3, 2]
        t1 = qf.Tensor(shape1, "Tensor 1")

        shape2 = [1, 4]
        t2 = qf.Tensor(shape2, "Tensor 2")

        with self.assertRaises(RuntimeError):
            t1.subtract(t2)
        

    def test_scale(self):

        shape = [3, 3]
        ref_arr = [ [0]*3 for i in range(3)]
        ref_arr1 = [ [0]*3 for i in range(3)]


        t1 = qf.Tensor(shape, "Tensor 1")
        t1.set([0, 0], 1.0)
        t1.set([1, 1], 2.0)
        t1.set([2, 2], 3.0)

        t1.scale(2)

        t2 = qf.Tensor(shape, "Tensor Ref")
        t2.set([0, 0], 2.0)
        t2.set([1, 1], 4.0)
        t2.set([2, 2], 6.0)

        # Converting data to array
        for i in range(t1.shape()[0]):
            for j in range(t1.shape()[1]):
                ref_arr[i][j] = t1.get([i, j])

        for i in range(t2.shape()[0]):
            for j in range(t2.shape()[1]):
                ref_arr1[i][j] = t2.get([i, j])      

          
        numpy.testing.assert_array_equal(ref_arr, ref_arr1)


    def test_scale2(self):

        shape = [4, 2]
        ref_arr = [ [0]*2 for i in range(4)]
        ref_arr1 = [ [0]*2 for i in range(4)]


        t1 = qf.Tensor(shape, "Tensor 1")
        t1.set([0, 0], 2.4)
        t1.set([1, 1], 1.1)
        t1.set([2, 1], 5.3)

        t1.scale(2.3)

        t2 = qf.Tensor(shape, "Tensor Ref")
        t2.set([0, 0], 5.52)
        t2.set([1, 1], 2.53)
        t2.set([2, 1], 12.19)

        # Converting data to array
        for i in range(t1.shape()[0]):
            for j in range(t1.shape()[1]):
                ref_arr[i][j] = t1.get([i, j])

        for i in range(t2.shape()[0]):
            for j in range(t2.shape()[1]):
                ref_arr1[i][j] = t2.get([i, j])      

          
        numpy.testing.assert_array_equal(ref_arr, ref_arr1)  

    def test_scale3(self):
        shape = [3, 3]
        ref_arr = [ [0]*3 for i in range(3)]
        ref_arr1 = [ [0]*3 for i in range(3)]


        t1 = qf.Tensor(shape, "Tensor 1")
        t1.set([0, 0], 1.0+1.0j)
        t1.set([1, 1], 2.0)
        t1.set([2, 2], 3.0)

        t1.scale(2)

        t2 = qf.Tensor(shape, "Tensor Ref")
        t2.set([0, 0], 2.0+2.0j)
        t2.set([1, 1], 4.0)
        t2.set([2, 2], 6.0)

        # Converting data to array
        for i in range(t1.shape()[0]):
            for j in range(t1.shape()[1]):
                ref_arr[i][j] = t1.get([i, j])

        for i in range(t2.shape()[0]):
            for j in range(t2.shape()[1]):
                ref_arr1[i][j] = t2.get([i, j])      

          
        numpy.testing.assert_array_equal(ref_arr, ref_arr1)   
    
    def test_scale4(self):

        shape = [10, 12]
        ref_arr = [ [0]*12 for i in range(10)]

        t1 = qf.Tensor(shape, "Tensor 1")

        random_array = np.random.rand(10, 12)
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                t1.set([i, j], random_array[i, j])

        t1.scale(2)

        for i in range(shape[0]):
            for j in range(shape[1]):
                ref_arr[i][j] = t1.get([i, j])

        random_array *= 2

        final_array = ref_arr - random_array

        self.assertLess(np.linalg.norm(final_array), 1e-16)


    def test_identity(self):

        shape = [2, 2]
        ref_arr = [ [0]*2 for i in range(2)]
        ref_arr1 = [ [0]*2 for i in range(2)]    

        t1 = qf.Tensor(shape, "Tensor 1")
        t1.set([0, 0], 2.4)
        t1.set([1, 1], 1.1)

        t2 = qf.Tensor(shape, "Tensor Ref")
        t2.set([0, 0], 1.0)
        t2.set([1, 1], 1.0)

        t1.identity()

        # Converting data to array
        for i in range(t1.shape()[0]):
            for j in range(t1.shape()[1]):
                ref_arr[i][j] = t1.get([i, j])

        for i in range(t2.shape()[0]):
            for j in range(t2.shape()[1]):
                ref_arr1[i][j] = t2.get([i, j]) 

        numpy.testing.assert_array_equal(ref_arr, ref_arr1)    

    def test_identity2(self):

        shape = [3, 3]
        ref_arr = [ [0]*3 for i in range(3)]
        ref_arr1 = [ [0]*3 for i in range(3)]
        
        t1 = qf.Tensor(shape, "Tensor 1")
        t1.set([0, 0], 1.0+1.0j)
        t1.set([1, 1], 1.1)
        t1.set([2, 2], 0.9)

        t2 = qf.Tensor(shape, "Tensor Ref")
        t2.set([0, 0], 1.0)
        t2.set([1, 1], 1.0)
        t2.set([2, 2], 1.0)

        t1.identity()

        # Converting data to array
        for i in range(t1.shape()[0]):
            for j in range(t1.shape()[1]):
                ref_arr[i][j] = t1.get([i, j])

        for i in range(t2.shape()[0]):
            for j in range(t2.shape()[1]):
                ref_arr1[i][j] = t2.get([i, j])

        final_arr = np.asarray(ref_arr) - np.asarray(ref_arr1)

        self.assertLess(np.linalg.norm(final_arr), 1e-16) 

    def test_identity3(self):

        shape = [15, 15]
        ref_arr = [ [0]*15 for i in range(15)]
        ref_arr1 = [ [0]*15 for i in range(15)]

        random_array = np.random.rand(15, 15)

        t1 = qf.Tensor(shape, "Tensor 1")

        for i in range(shape[0]):
            for j in range(shape[1]):
                t1.set([i, j], random_array[i, j])

        for i in range(shape[0]):
            ref_arr1[i][i] = 1.0
        
        t1.identity()

        for i in range(shape[0]):
            for j in range(shape[1]):
                ref_arr[i][j] = t1.get([i, j])

        final_arr = np.asarray(ref_arr) - np.asarray(ref_arr1)

        self.assertLess(np.linalg.norm(final_arr), 1e-16)
        


    def test_identity4(self):

        shape = [3, 3, 2]

        t1 = qf.Tensor(shape, "Tensor 1")
        t1.set([0, 0, 0], 2.4)
        t1.set([1, 1, 1], 1.1)
        t1.set([2, 2, 1], 0.9)

        with self.assertRaises(RuntimeError):
            t1.identity()

    def test_zero(self):

        shape = [12, 11]
        ref_arr = [ [0]*11 for i in range(12)]

        t1 = qf.Tensor(shape, "Tensor 1")

        random_array = np.random.rand(12, 11)

        for i in range(shape[0]):
            for j in range(shape[1]):
                t1.set([i, j], random_array[i, j])

        t1.zero()

        for i in range(shape[0]):
            for j in range(shape[1]):
                ref_arr[i][j] = t1.get([i, j])

        self.assertEqual(np.linalg.norm(np.asarray(ref_arr)), 0)

    def test_zero2(self):
        
        shape = [12, 11]
        ref_arr = [ [0]*11 for i in range(12)]

        t1 = qf.Tensor(shape, "Tensor 1")

        random_array = np.random.rand(12, 11)

        for i in range(shape[0]):
            for j in range(shape[1]):
                t1.set([i, j], random_array[i, j])

        t1.set([2, 5], 2.0+2.0j)

        t1.zero()

        for i in range(shape[0]):
            for j in range(shape[1]):
                ref_arr[i][j] = t1.get([i, j])

        self.assertEqual(np.linalg.norm(np.asarray(ref_arr)), 0)

    def test_symmetrize(self):

        shape = [12, 12]
        ref_arr = [ [0]*12 for i in range(12)]

        t1 = qf.Tensor(shape, "Tensor 1")

        random_array = np.random.rand(12, 12)

        for i in range(shape[0]):
            for j in range(shape[1]):
                t1.set([i, j], random_array[i, j])

        random_array = (random_array + random_array.T)/2
        
        t1.symmetrize()

        for i in range(shape[0]):
            for j in range(shape[1]):
                ref_arr[i][j] = t1.get([i, j])

        final_arr = np.asarray(ref_arr) - np.asarray(random_array)

        self.assertLess(np.linalg.norm(final_arr), 1e-16)

    def test_symmetrize2(self):

        shape = [12, 12]
        ref_arr = [ [0]*12 for i in range(12)]

        t1 = qf.Tensor(shape, "Tensor 1")

        random_array = np.random.rand(12, 12)
        random = np.array(random_array, dtype = np.dtype(np.complex128))

        for i in range(shape[0]):
            for j in range(shape[1]):
                t1.set([i, j], random_array[i, j])

        t1.set([5, 3], 2.0+2.0j)
        random[5, 3] = 2.0+2.0j

        random = (random + random.T)/2
        
        t1.symmetrize()

        for i in range(shape[0]):
            for j in range(shape[1]):
                ref_arr[i][j] = t1.get([i, j])

        final_arr = np.asarray(ref_arr) - np.asarray(random)

        self.assertLess(np.linalg.norm(final_arr), 1e-16)

    def test_symmetrize3(self):
        shape = [4, 7]
        t1 = qf.Tensor(shape, "Tensor 1")

        with self.assertRaises(RuntimeError):
            t1.symmetrize()

    def test_antisymmetrize(self):

        shape = [12, 12]
        ref_arr = [ [0]*12 for i in range(12)]

        t1 = qf.Tensor(shape, "Tensor 1")

        random_array = np.random.rand(12, 12)

        for i in range(shape[0]):
            for j in range(shape[1]):
                t1.set([i, j], random_array[i, j])

        random_array = (random_array - random_array.T)/2
        
        t1.antisymmetrize()

        for i in range(shape[0]):
            for j in range(shape[1]):
                ref_arr[i][j] = t1.get([i, j])

        final_arr = np.asarray(ref_arr) - np.asarray(random_array)

        self.assertLess(np.linalg.norm(final_arr), 1e-16)        

    def test_antisymmetrize2(self):

        shape = [12, 12]
        ref_arr = [ [0]*12 for i in range(12)]

        t1 = qf.Tensor(shape, "Tensor 1")

        random_array = np.random.rand(12, 12)
        random = np.array(random_array, dtype = np.dtype(np.complex128))

        for i in range(shape[0]):
            for j in range(shape[1]):
                t1.set([i, j], random_array[i, j])

        t1.set([5, 3], 2.0+2.0j)
        random[5, 3] = 2.0+2.0j

        random = (random - random.T)/2
        
        t1.antisymmetrize()

        for i in range(shape[0]):
            for j in range(shape[1]):
                ref_arr[i][j] = t1.get([i, j])

        final_arr = np.asarray(ref_arr) - np.asarray(random)

        self.assertLess(np.linalg.norm(final_arr), 1e-16)

    def test_antisymmetrize3(self):
        shape = [9, 1]
        t1 = qf.Tensor(shape, "Tensor 1")

        with self.assertRaises(RuntimeError):
            t1.antisymmetrize()

    def test_transpose(self):

        shape = [13, 13]
        ref_arr = [ [0]*13 for i in range(13)]

        t1 = qf.Tensor(shape, "Tensor 1")

        random_array = np.random.rand(13, 13)

        for i in range(shape[0]):
            for j in range(shape[1]):
                t1.set([i, j], random_array[i, j])
        
        t2 = t1.transpose()

        for i in range(shape[0]):
            for j in range(shape[1]):
                ref_arr[i][j] = t2.get([i, j])
        
        final_arr = np.asarray(random_array.T) - np.asarray(ref_arr)

        self.assertLess(np.linalg.norm(final_arr), 1e-16)

    def test_transpose2(self):
        
        shape = [13, 13]
        ref_arr = [ [0]*13 for i in range(13)]

        t1 = qf.Tensor(shape, "Tensor 1")

        random_array = np.random.rand(13, 13)
        random = np.array(random_array, dtype = np.dtype(np.complex128))
        random[4, 9] = 2.0+2.0j

        for i in range(shape[0]):
            for j in range(shape[1]):
                t1.set([i, j], random[i, j])
        
        t2 = t1.transpose()

        for i in range(shape[0]):
            for j in range(shape[1]):
                ref_arr[i][j] = t2.get([i, j])

        
        
        final_arr = np.asarray(random.T) - np.asarray(ref_arr)

        self.assertLess(np.linalg.norm(final_arr), 1e-16)

    def test_transpose3(self):

        shape = [3, 3, 3]
        t1 = qf.Tensor(shape, "Tensor 1")

        with self.assertRaises(RuntimeError):
            t1.transpose()

    def test_general_transpose(self):

        shape = [13, 13, 13]
        axes = [1, 0, 2]
        ref_arr = [[[0 for i in range(13)] for j in range(13)] for k in range(13)]

        t1 = qf.Tensor(shape, "Tensor 1")

        random_array = np.random.rand(13, 13, 13)

        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    t1.set([i, j, k], random_array[i, j, k])
        
        t2 = t1.general_transpose(axes)
        random = random_array.transpose(axes)

        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    ref_arr[i][j][k] = t2.get([i, j, k])
            

        
        final_arr = np.asarray(random) - np.asarray(ref_arr)

        self.assertLess(np.linalg.norm(final_arr), 1e-16)

    def test_general_transpose2(self):
        shape = [13, 13, 13]
        axes = [1, 0, 2]
        ref_arr = [[[0 for i in range(13)] for j in range(13)] for k in range(13)]

        t1 = qf.Tensor(shape, "Tensor 1")

        random_array = np.random.rand(13, 13, 13)
        random = np.array(random_array, dtype = np.dtype(np.complex128))
        random[4, 9, 3] = 2.0+2.0j

        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    t1.set([i, j, k], random[i, j, k])
        
        t2 = t1.general_transpose(axes)

        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    ref_arr[i][j][k] = t2.get([i, j, k])

        final_arr = np.asarray(random.transpose(axes)) - np.asarray(ref_arr)

        self.assertLess(np.linalg.norm(final_arr), 1e-16)        

    def test_general_transpose3(self):

        shape = [4, 4, 4]
        axes = [2, 1]

        t1 = qf.Tensor(shape, "Tensor 1")

        with self.assertRaises(ValueError):
            t1.general_transpose(axes)

    def test_fill_from_np(self):

        shape = [3, 4, 4]
        ref_arr = [[[0 for i in range(4)] for j in range(4)] for k in range(3)]


        t1 = qf.Tensor(shape, "Tensor 1")

        random_array = np.random.rand(3, 4, 4)
        random = np.array(random_array, dtype = np.dtype(np.complex128))

        t1.fill_from_nparray(random.ravel(), shape)

        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    ref_arr[i][j][k] = t1.get([i, j, k])

        final_arr = np.asarray(ref_arr) - np.asarray(random)

        self.assertLess(np.linalg.norm(final_arr), 1e-16)

    def test_fill_from_np2(self):

        shape = [16, 15, 12, 9]
        ref_arr = [[[[0 for i in range(9)] for j in range(12)] for k in range(15)] for z in range(16)]


        t1 = qf.Tensor(shape, "Tensor 1")

        random_array = np.random.rand(16, 15, 12, 9)
        random = np.array(random_array, dtype = np.dtype(np.complex128))

        t1.fill_from_nparray(random.ravel(), shape)

        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    for z in range(shape[3]):
                        ref_arr[i][j][k][z] = t1.get([i, j, k, z])

        final_arr = np.asarray(ref_arr) - np.asarray(random)

        self.assertLess(np.linalg.norm(final_arr), 1e-16)


    def test_fill_from_np3(self):

        shape1 = [2, 4]
        shape2 = [3, 5]

        t1 = qf.Tensor(shape1, "Tensor 1")

        random_array = np.random.rand(2, 4)
        random = np.array(random_array, dtype = np.dtype(np.complex128))


        with self.assertRaises(RuntimeError):
            t1.fill_from_nparray(random.ravel(), shape2)

    def test_norm(self):

        shape1 = [4, 5]

        t1 = qf.Tensor(shape1, "Tensor 1")

        random_array = np.random.rand(4, 5)
        random = np.array(random_array, dtype = np.dtype(np.complex128))

        random[2, 3] = 3.0 + 2.5j 

        t1.fill_from_nparray(random.ravel(), shape1)
        qf_result = t1.norm()
        ref = np.linalg.norm(random)

        diff = qf_result - ref

        self.assertLess(abs(diff), 1e-14)

    def test_norm2(self):

        shape1 = [12, 15]

        t1 = qf.Tensor(shape1, "Tensor 1")

        random_array = np.random.rand(12, 15)
        random = np.array(random_array, dtype = np.dtype(np.complex128))

        random[3, 10] = 12.2 + 2.6j

        t1.fill_from_nparray(random.ravel(), shape1)
        qf_result = t1.norm()
        ref = np.linalg.norm(random)

        diff = qf_result - ref

        self.assertLess(abs(diff), 1e-14)






unittest.main()


"""

Questions:

what happens if you add a 2x3 tensor to a 3x2 tensor
or a 4x1 tensor to a 2x2 tensor

what part of shape is the x and y grid?
    is [4, 2] a grid of 4 rows of 2 wide?


add more tests
complex numbers
tensor to np helper function
rank component grouping function


"""



