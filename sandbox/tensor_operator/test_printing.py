import qforte as qf

dim = 2
max_nbody = 1
is_spatial = False
is_restricted = False

Top = qf.TensorOperator(
    max_nbody = max_nbody, 
    dim = dim
    )

print(Top)