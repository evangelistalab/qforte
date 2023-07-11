import qforte as qf
import numpy as np

# Set the seed value
seed_value = 42
np.random.seed(seed_value)

# Generate a random NumPy array
array_shape = (4, 4)  # Specify the shape of the array
random_array = np.random.rand(*array_shape)

shape1 = [4, 4]
axes = [1, 0]

# ary = np.zeros(shape=shape1)
# ary[0,1] = -4.0
# ary[3,0] = -2.0
# print("\n")
# print(ary)
# print("\n")

aryT = random_array.transpose(axes)
print("\n")
print(aryT)
print("\n")

T1 = qf.Tensor(shape=shape1, name='steve')
for i in range(shape1[0]):
    for j in range(shape1[1]):
        T1.set([i,j], random_array[i,j])

T2 = T1.transpose()


T3 = T1.general_transpose(axes)
# T2.set_name('transposed')

print(T1)
print(T2)
print(T3)

