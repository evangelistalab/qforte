import qforte as qf

print("\n Initial FCIcomp Stuff")
print("===========================")
nel = 4
sz = 0
norb = 4

fci_comp = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fci_comp.hartree_fock()

print(fci_comp.str(print_data=True))

dim = 2*norb
max_nbody = 2

Top = qf.TensorOperator(
    max_nbody = max_nbody, 
    dim = dim
    )

print("\n SQOP Stuff")
print("===========================")
sqop1 = qf.SQOperator()
sqop2 = qf.SQOperator()
sqop1.add_term(2.0, [5], [1])
sqop1.add_term(2.0, [1], [5])

sqop1.add_term(2.0, [4], [0])
sqop1.add_term(2.0, [0], [4])

sqop2.add_term(4.5, [5, 4], [1, 0])
sqop2.add_term(4.5, [0, 1], [4, 5])

print(sqop1)
print(sqop2)

# so all seems to be working except 
print("\n Tensor Stuff")
print("===========================")
print(f"Ranks present2: {sqop2.ranks_present()}")

Top.add_sqop_of_rank(sqop1, 2)
Top.add_sqop_of_rank(sqop2, 4)

[H0, H1, H2] = Top.tensors()

# print(H1)

# fci_comp.apply_tensor_spin_1bdy(H1, norb)

# print("\n Final FCIcomp 1-body Stuff")
# print("================================")
# print(fci_comp)


fci_comp.hartree_fock()
fci_comp.apply_tensor_spin_12bdy(H1, H2, norb)

print("\n Final FCIcomp 12-body Stuff")
print("=================================")
print(fci_comp)