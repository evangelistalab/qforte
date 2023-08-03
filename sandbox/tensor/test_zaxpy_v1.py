import qforte as qf

shape1 = [4, 4]

T1 = qf.Tensor(shape=shape1, name='ZAXPY OUT')
T1.set([0,0], 2.0)
T1.set([1,1], 2.0)
T1.set([2,0], 2.0)

T2 = qf.Tensor(shape=shape1, name='bob')
T2.set([0,0], 2.0)
T2.set([1,1], 2.0)
T2.set([0,2], -2.0)

print(T1)
print(T2)

T1.zaxpy(T2, 1.0)

print(T1)
