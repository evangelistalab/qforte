import numpy as np
import qforte as qf
import re 

# Create two 2-dimensional arrays
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Perform matrix multiplication using einsum
result = np.einsum('ij,ji->i', A, B)

print(result)

shape1 = [2, 2]

T1 = qf.Tensor(shape=shape1, name='steve')
T1.set([0,0], 1.0)
T1.set([0,1], 2.0)
T1.set([1,0], 5.0)
T1.set([1,1], 6.0)

T2 = qf.Tensor(shape=shape1, name='bob')
T2.set([0,0], 3.0)
T2.set([0,1], 4.0)
T2.set([1,0], 7.0)
T2.set([1,1], 8.0)

print(T1)
print(T2)

my_einstr = "ij,ji->i"
Astr, Bstr, Cstr = re.split(r',|->', my_einstr)
print(f"{Astr}, {Bstr}, {Cstr}")

Tcontainer = qf.Tensor(shape=[2], name='Tcontainer')
Tcontainer.set([0], 6.5)
Tcontainer.set([1], 4.5)

print(Tcontainer)

qf.Tensor.einsum(
    [x for x in Astr], 
    [x for x in Bstr], 
    [x for x in Cstr], 
    T1, 
    T2,
    Tcontainer,
    1.0,
    0.0)

# print(T4)
print(Tcontainer)

