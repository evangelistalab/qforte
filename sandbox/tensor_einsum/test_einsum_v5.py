import numpy as np
import qforte as qf
import re 

def create_random_array(shape, seed=None):
    if seed is not None:
        np.random.seed(seed)

    array = np.random.rand(*shape)

    #  # Get the indices of the upper triangle for a flattened array
    # triu_indices = np.triu_indices(np.prod(shape), k=1)

    # # Convert the flat indices to the indices in the original shape
    # flat_index_rows, flat_index_cols = triu_indices
    # index_rows, index_cols = np.unravel_index(flat_index_rows, shape), np.unravel_index(flat_index_cols, shape)

    # # Set the values in the lower triangle equal to the values in the upper triangle
    # array[index_rows, index_cols] = array[index_cols, index_rows]

    return array

dim1 = 4

shape1 = [dim1, dim1, dim1, dim1]
# shape1 = [dim1, dim1]
seed_value = 42

R1 = create_random_array(shape1, seed=seed_value)
R2 = create_random_array(shape1, seed=seed_value+1)
R1 = R1.astype(np.complex128)
R2 = R2.astype(np.complex128)

# Perform matrix multiplication using einsum
einstr   = 'pqrs,qrst->pqrt'  # NOPE goes to: C[qrpt] = A[qrps] * B[qrst]
einstr2  = 'qrps,qrst->qrpt'  # This works
# einstr  = 'qrps,qrst->qrpt'  # Now this works!

# in principal we could use the permute function and then
# call the funciton the way it wants?? 


# einstr  = 'pqrs,pqrs->pqsr'  # NOPE
# einstr  = 'pqrs,pqrs->pqrs'  # works (tensor dot?)

# einstr  = 'ij,ji->ij'  # Does not work, goes to C[ij] = A[ij] * B[ij]
# einstr  = 'ij,ij->ij'  # So try this

# gives following output
# Original: C[ij] = A[ij] * B[ji]
# New:      C[ij] = A[ij] * B[ij] # note this has changed, and we don't want that...
# A viable solution is to Transpose B2 mid function call

# C Permuted: No
# A Permuted: No
# B Permuted: Yes
# C Transposed: No
# A Transposed: No
# B Transposed: Yes

# einstr  = 'ij,ij->ij'  # Works???
# einstr  = 'ij,jk->ik'  # Works???
# einstr  = 'ij,jk->ki'  # Works???

# einstr  = 'ji,jk->ki'  # Works???
# einstr  = 'ji,kj->ki'  # Works???
# einstr  = 'ji,kj->ik'  # Works???

# currently is working with only regular ordering!
print(f"einstring: {einstr}")
C2 = np.einsum(einstr, R1, R2)
# print(f"C = einsum(..., A, B) \n {C2} \n")


T1 = qf.Tensor(shape=shape1, name='T1')
T2 = qf.Tensor(shape=shape1, name='T2')
T1.fill_from_np(R1.ravel(), shape1)
T2.fill_from_np(R2.ravel(), shape1)

# print(T1)

# 'pqrs,qrst->pqrt'  # NOPE goes to: C[qrpt] = A[qrps] * B[qrst]
# 'qrps,qrst->qrpt'

# T1 ==> (pqrs->qrps) axis1 = [0,1,2,3] axis2 = [1,2,0,3]
# T2 ==> (qrst->qrst) No Transpose
# Tc ==> (pqrt->qrpt) axis1 = [0,1,2,3] axis2 = [1,2,0,3]

# works for the rank-2 case!
T1 = T1.general_transpose([2,0,1,3]) 

# now try the more complicated rank-4 case

Astr, Bstr, Cstr = re.split(r',|->', einstr)
print(f"{Astr}, {Bstr}, {Cstr}")

Tcontainer = qf.Tensor(shape=C2.shape, name='Tcontainer')

qf.Tensor.einsum(
    [x for x in Astr], 
    [x for x in Bstr], 
    [x for x in Cstr], 
    T1, 
    T2, # swapping seems to fix???
    Tcontainer,
    1.0, 
    0.0) 

# Tcontainer = Tcontainer.general_transpose([2,0,1,3])

print(Tcontainer)
print(f"C = einsum(..., A, B) \n {C2} \n")

Tdiff = qf.Tensor(shape=C2.shape, name='Tdiff')
Tdiff.fill_from_np(C2.ravel(), C2.shape)
Tdiff.scale(-1.0)
Tdiff.add(Tcontainer)
print(Tdiff)



