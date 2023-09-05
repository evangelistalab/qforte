import qforte as qf
 
import time
 
print("\n Initial FCIcomp Stuff")
print("===========================")
nel = 6
sz = 0
norb = 6
 
fci_comp = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fci_comp.hartree_fock()
 
fock_comp = qf.Computer(norb * 2)
 
# print(fci_comp.str(print_data=True))
 
dim = 2*norb
max_nbody = 1
 
Top = qf.TensorOperator(
    max_nbody = max_nbody,
    dim = dim
    )
 
print("\n SQOP Stuff")
print("===========================")
sqop = qf.SQOperator()
 
# sqop.add_term(2.0, [5], [1])
# sqop.add_term(2.0, [1], [5])
 
# sqop.add_term(2.0, [4], [0])
# sqop.add_term(2.0, [0], [4])
 
for i in range(norb*2):
    for j in range(norb*2):
        sqop.add_term(2.0+i-j, [i], [j])
        sqop.add_term(2.0+i-j, [j], [i])
 
# print(sqop)
 
t_ap1bdy_fock = time.perf_counter()
fock_comp.apply_sq_operator(sqop)
t_ap1bdy_fock = time.perf_counter() - t_ap1bdy_fock
 
# so all seems to be working except
print("\n Tensor Stuff")
print("===========================")
Top.add_sqop_of_rank(sqop, sqop.ranks_present()[0])
 
[H0, H1] = Top.tensors()
 
# print(H1)
 
t_ap1bdy_fci = time.perf_counter()
fci_comp.apply_tensor_spin_1bdy(H1, norb)
t_ap1bdy_fci = time.perf_counter() - t_ap1bdy_fci
 
# print("\n Final FCIcomp Stuff")
# print("===========================")
# print(fci_comp)
 
print("\n Timing")
print("======================================================")
print(f" nqbit:     {norb*2}")
print(f" fci_time:  {t_ap1bdy_fci:12.8f}")
print(f" fock_time: {t_ap1bdy_fock:12.8f}")
print(f" speedup:   {(t_ap1bdy_fock/t_ap1bdy_fci):12.8f}")