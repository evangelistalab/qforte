import qforte as qf

shape1 = [4, 4, 3]

T1 = qf.Tensor()

print(T1)

T1.zero_with_shape(shape1)

print(T1)
# LGTM