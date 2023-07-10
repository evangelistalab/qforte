import qforte as qf

shape1 = [4, 4]

T1 = qf.Tensor(shape=shape1, name='steve')
print(T1)

T1.identity()

# LGTM(Nick)
print(T1)
