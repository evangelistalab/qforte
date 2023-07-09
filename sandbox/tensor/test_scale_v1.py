import qforte as qf

shape1 = [4, 4]

T1 = qf.Tensor(shape=shape1, name='steve')
T1.set([0,0], 2.0)
T1.set([1,1], 2.0)
T1.set([2,0], 2.0)

print(T1)

T1.scale(1.0/3.0)

# LGTM(Nick)
print(T1)
