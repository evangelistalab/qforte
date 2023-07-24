import numpy as np
import qforte as qf
import re 

# Create two 2-dimensional arrays
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Perform matrix multiplication using einsum
result = np.einsum('ij,ji->i', A, B)

print(result)

shape1 = [3, 3]

T1 = qf.Tensor(shape=shape1, name='steve')
T1.set([0,0], 1.0)
T1.set([0,1], 2.0)
T1.set([0,2], 3.0)
T1.set([1,0], 4.0)
T1.set([1,1], 5.0)
T1.set([1,2], 6.0)
T1.set([2,0], 7.0)
T1.set([2,1], 8.0)
T1.set([2,2], 9.0)

# T2 = qf.Tensor(shape=shape1, name='bob')
# T2.set([0,0], 3.0)
# T2.set([0,1], 4.0)
# T2.set([1,0], 7.0)
# T2.set([1,1], 8.0)

T2 = T1 # SHALLOW COPY!!

T2.set([2,2], 7.5)

T3 = qf.Tensor(shape=[3], name='joe')

T3.set([0], 1.0)
# T3 = qf.Tensor()

print(T1)
print(T2)

my_einstr = "ij,ji->i"

Astr, Bstr, Cstr = re.split(r',|->', my_einstr)

print(f"{Astr}, {Bstr}, {Cstr}")

T4 = qf.Tensor.einsum(
    [x for x in Astr], 
    [x for x in Bstr], 
    [x for x in Cstr], 
    T1, 
    T2,
    T3,
    1.0,
    1.0)

# print(T4)
print(T4)

