import qforte as qf

nel = 4
sz = 0
norb = 4

fci_comp = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fci_comp.hartree_fock()

print(fci_comp.str(print_data=True))

dim = 2*norb
max_nbody = 1

# Top = qf.TensorOperator(
#     max_nbody = max_nbody, 
#     dim = dim
#     )

print("\n SQOP Stuff")
print("===========================")
sqop = qf.SQOperator()

sqop.add_term(1.6, [], [])

# beta
sqop.add_term(3.0, [5], [1])
sqop.add_term(3.0, [1], [5])

# alfa
sqop.add_term(2.0, [4], [0])
sqop.add_term(2.0, [0], [4])

# alfa alfa
sqop.add_term(4.5, [6, 4], [2, 0])
sqop.add_term(4.5, [0, 2], [4, 6])

# alfa beta
sqop.add_term(5.5, [4, 5], [1, 0])
sqop.add_term(5.5, [1, 0], [5, 4])

#  Bad SQOP
# ======================================================
#  -0.104645 ( 7^ 6^ 3 2 )


print(sqop)


# so all seems to be working except 
# print("\n Tensor Stuff")
# print("===========================")
# Top.add_sqop_of_rank(sqop, sqop.ranks_present()[0])

# [H0, H1] = Top.tensors()

# print(H1)

print("\n Initial FCIcomp Stuff")
print("===========================")
print(fci_comp)

fci_comp.apply_sqop(sqop)

print("\n Final FCIcomp Stuff")
print("===========================")
print(fci_comp)