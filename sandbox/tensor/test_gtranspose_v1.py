import qforte as qf
import numpy as np

shape1 = [4, 4]

# map which axis to which axis I think
# so [0, 1] won't do anyting
# [1, 0] with do a regular matrix transpose
axes = [1, 0]

ary = np.zeros(shape=shape1)
ary[0,1] = -4.0
ary[3,0] = -2.0
print("\n")
print(ary)
print("\n")

aryT = ary.transpose(axes)
print("\n")
print(aryT)
print("\n")

T1 = qf.Tensor(shape=shape1, name='steve')
T1.set([0,1], -4.0)
T1.set([3,0], -2.0)

T2 = T1.transpose()


T3 = T1.general_transpose(axes)
# T2.set_name('transposed')

print(T1)
print(T2)
print(T3)

