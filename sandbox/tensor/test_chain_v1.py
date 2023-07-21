import qforte as qf

shape1 = [4, 4]

T1 = qf.Tensor(shape=shape1, name='steve')
T1.set([0,0], 2.0)
T1.set([1,1], 2.0)
T1.set([2,0], 2.0)

T2 = qf.Tensor(shape=shape1, name='bob')
T2.identity()

print(T1)
print(T2)

T3 = qf.Tensor.chain([T1, T2], [False, False])

# LGTM!
print(T3)
