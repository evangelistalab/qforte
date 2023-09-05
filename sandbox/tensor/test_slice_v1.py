import qforte as qf

shape = [5, 5]

t1 = qf.Tensor(shape, "Tensor 1")

for i in range(shape[0]):
    for j in range(shape[1]):
        t1.set([i, j], i + j)

print(t1)

            # (include, exclude)

t2 = t1.slice([(2, 3), (2, 3)])

print(t2)