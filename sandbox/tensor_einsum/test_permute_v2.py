import qforte as qf
import numpy as np
import re

# Set the seed value
seed_value = 42
np.random.seed(seed_value)

# Generate a random NumPy array
array_shape = (2, 2, 2, 2)  # Specify the shape of the array
random_array = np.random.rand(*array_shape)

shape1 = [2, 2, 2, 2]
axes = [0, 2, 1, 3]

# random_arrayT = random_array.transpose(axes)
random_arrayT = np.moveaxis(random_array, 1, 2) * (-1.0)
print("\n")
print(random_arrayT)
print("\n")

T1 = qf.Tensor(shape=shape1, name='steve')
for i in range(shape1[0]):
    for j in range(shape1[1]):
        for k in range(shape1[2]):
            for l in range(shape1[3]):
                T1.set([i,j,k,l], random_array[i,j,k,l])

T3 = T1.general_transpose(axes)

# print(T3)

# want to use permute to do this same operation 
permstr = 'ijkl->ikjl'
Astr, Cstr = re.split(r'->', permstr)
Tcontainer = qf.Tensor(shape=T3.shape(), name='Tcontainer')
print(f"{Astr}, {Cstr}")

qf.Tensor.permute(
    [x for x in Astr],  
    [x for x in Cstr], 
    T1, 
    Tcontainer,
    1.0,
    0.0)

Tcontainer.scale(-1.0)

print(Tcontainer)

