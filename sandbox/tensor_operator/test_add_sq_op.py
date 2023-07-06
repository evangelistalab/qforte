import qforte as qf

dim = 4
max_nbody = 1
is_spatial = False
is_restricted = False

Top = qf.TensorOperator(
    max_nbody = max_nbody, 
    dim = dim
    )

sqop = qf.SQOperator()
sqop.add_term(1.0, [2], [0])

print(sqop.ranks_present())

print(sqop)
print(Top)

Top.add_sqop_of_rank(sqop, 2)