import qforte as qf
import numpy as np

# Set the seed value
seed_value = 42
np.random.seed(seed_value)

# Generate a random NumPy array
array_shape = (4, 3, 2)  # Specify the shape of the array
random_array = np.random.rand(*array_shape)

shape1 = [4, 3, 2]
axes = [0, 2, 1]

random_arrayT = random_array.transpose(axes)
print("\n")
print(random_arrayT)
print("\n")

T1 = qf.Tensor(shape=shape1, name='steve')
for i in range(shape1[0]):
    for j in range(shape1[1]):
        for k in range(shape1[2]):
            T1.set([i,j, k], random_array[i,j,k])

T3 = T1.general_transpose(axes)

print(T3)

