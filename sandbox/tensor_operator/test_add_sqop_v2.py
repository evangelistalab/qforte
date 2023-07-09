import qforte as qf

dim = 4
max_nbody = 2
is_spatial = False
is_restricted = False

Top = qf.TensorOperator(
    max_nbody = max_nbody, 
    dim = dim
    )

sqop = qf.SQOperator()
sqop.add_term(6.9, [3, 2], [0, 1])
sqop.add_term(6.9, [1, 0], [2, 3])

print(f"sqop.ranks_present(): {sqop.ranks_present()}")

ablk, bblk = sqop.get_largest_alfa_beta_indices()
print(f"ablk: {ablk}")
print(f"bblk: {bblk}")

print(sqop)

print("\n\n")

# so all seems to be working except 
print("Now printing qf Debug Stuff")
print("===========================")
Top.add_sqop_of_rank(sqop, sqop.ranks_present()[0])

# print(Top)