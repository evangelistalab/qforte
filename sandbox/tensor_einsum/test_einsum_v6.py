import numpy as np
import qforte as qf
import re 

def create_random_array(shape, seed=None):
    if seed is not None:
        np.random.seed(seed)

    array = np.random.rand(*shape)

    return array

dim1 = 4

# shape1 = [dim1, dim1, dim1, dim1]
shape1 = [dim1, dim1]
seed_value = 42

R1 = create_random_array(shape1, seed=seed_value)
R2 = create_random_array(shape1, seed=seed_value+1)
R1 = R1.astype(np.complex128)
R2 = R2.astype(np.complex128)

# Perform matrix multiplication using einsum
# einstr   = 'pqrs,qrst->pqrt'  # Works!
# einstr   = 'qrps,qrst->qrpt'  
# einstr   = 'qprs,psqr->p'  

# einstr   = 'pppp,pppp->p'  #does not work
# einstr   = 'pqpq,pqpq->p'  #does not work
# einstr   = 'pqpq,pqpq->p'  #does not work
# einstr   = 'pqrs,pqrs->sr' #works!
# einstr   = 'pqrs,pqrs->s'  #works!

einstr   = 'ij,jk->ik'  #works!



# may still be some debugging but ok for now...

print(f"einstring: {einstr}")
C2 = np.einsum(einstr, R1, R2)

T1 = qf.Tensor(shape=shape1, name='T1')
T2 = qf.Tensor(shape=shape1, name='T2')
T1.fill_from_np(R1.ravel(), shape1)
T2.fill_from_np(R2.ravel(), shape1)

Astr, Bstr, Cstr = re.split(r',|->', einstr)
print(f"{Astr}, {Bstr}, {Cstr}")

Tcontainer = qf.Tensor(shape=C2.shape, name='Tcontainer')

qf.Tensor.einsum(
    [x for x in Astr], 
    [x for x in Bstr], 
    [x for x in Cstr], 
    T1, 
    T2, 
    Tcontainer,
    1.0, 
    0.0) 

print(Tcontainer)
print(f"C = einsum(..., A, B) \n {C2} \n")

Tdiff = qf.Tensor(shape=C2.shape, name='Tdiff')
Tdiff.fill_from_np(C2.ravel(), C2.shape)
Tdiff.scale(-1.0)
Tdiff.add(Tcontainer)
print(Tdiff)



