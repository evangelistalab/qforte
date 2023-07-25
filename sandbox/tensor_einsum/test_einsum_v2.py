import numpy as np
import qforte as qf
import re 

# Create two 2-dimensional arrays
A = np.array([[1, 2, 3], [4, 5,   6], [7, 8, 9]])
B = np.array([[1, 2, 3], [4, 5.5, 6], [7, 8, 9]])

# Perform matrix multiplication using einsum
# einstr  = 'ij,ji->i'
# einstr  = 'ij,ij->i'
einstr  = 'ij,jk->ik'
# einstr  = 'ij,kj->ik'

# currently is working with only regular ordering!

result = np.einsum(einstr, A, B)

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

T2 = qf.Tensor(shape=shape1, name='steve')
T2.set([0,0], 1.0)
T2.set([0,1], 2.0)
T2.set([0,2], 3.0)
T2.set([1,0], 4.0)
T2.set([1,1], 5.5)
T2.set([1,2], 6.0)
T2.set([2,0], 7.0)
T2.set([2,1], 8.0)
T2.set([2,2], 9.0)

# my_einstr = "ij,ji->i"
Astr, Bstr, Cstr = re.split(r',|->', einstr)
print(f"{Astr}, {Bstr}, {Cstr}")

# Tcontainer = qf.Tensor(shape=[3,3], name='Tcontainer')
Tcontainer = qf.Tensor(shape=[3,3], name='Tcontainer')

# Tcontainer.set([0], 6.5)
# Tcontainer.set([1], 4.5)
# Tcontainer.set([2], 3.5)

# print(Tcontainer)

qf.Tensor.einsum(
    [x for x in Astr], 
    [x for x in Bstr], 
    [x for x in Cstr], 
    T1, 
    T2,
    Tcontainer,
    1.0, # does nothing!
    0.0) # does nothing

# print(T4)
print(Tcontainer)

