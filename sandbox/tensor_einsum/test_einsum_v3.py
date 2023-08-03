import numpy as np
import qforte as qf
import re 

# Create two 2-dimensional arrays
A = np.array([[1, 2, 3], 
              [4, 5, 6], 
              [7, 8, 9]])

B = np.array([[1.5, 2.5, 3.5], 
              [4.5, 5.5, 6.5], 
              [7.5, 8.5, 9.5]])

# C1 = A @ B
# print(f"C = A@B \n {C1} \n")

# Perform matrix multiplication using einsum
# einstr  = 'ij,ji->i'  # NOPE
# einstr  = 'ij,ij->i'  # works and returns same result as above
# einstr  = 'ij,jk->ik' # works
# einstr  = 'ij,jk->ki' # works!
einstr  = 'ij,kj->ik'   # works!

# currently is working with only regular ordering!
print(f"einstring: {einstr}")
C2 = np.einsum(einstr, A, B)
print(f"C = einsum(..., A, B) \n {C2} \n")

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
T2.set([0,0], 1.5)
T2.set([0,1], 2.5)
T2.set([0,2], 3.5)
T2.set([1,0], 4.5)
T2.set([1,1], 5.5)
T2.set([1,2], 6.5)
T2.set([2,0], 7.5)
T2.set([2,1], 8.5)
T2.set([2,2], 9.5)

Astr, Bstr, Cstr = re.split(r',|->', einstr)
print(f"{Astr}, {Bstr}, {Cstr}")

Tcontainer = qf.Tensor(shape=[3,3], name='Tcontainer')
# Tcontainer = qf.Tensor(shape=[3], name='Tcontainer')

qf.Tensor.einsum(
    [x for x in Astr], 
    [x for x in Bstr], 
    [x for x in Cstr], 
    T1, 
    T2, # swapping seems to fix???
    Tcontainer,
    1.0, # does nothing!
    0.0) # does nothing


print(Tcontainer)

