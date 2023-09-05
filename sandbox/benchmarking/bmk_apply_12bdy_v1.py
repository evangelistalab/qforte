import qforte as qf
 
import time

# Define the reference and geometry lists.
geom = [
    ('H', (0., 0., 1.0)), 
    ('H', (0., 0., 2.0)),
    ('H', (0., 0., 3.0)), 
    ('H', (0., 0., 4.0)),
    ('H', (0., 0., 5.0)), 
    ('H', (0., 0., 6.0)),
    ('H', (0., 0., 7.0)), 
    ('H', (0., 0., 8.0)),
    # ('H', (0., 0., 9.0)), 
    # ('H', (0., 0.,10.0))
    ]

# Get the molecule object that now contains both the fermionic and qubit Hamiltonians.
mol = qf.system_factory(build_type='psi4', mol_geometry=geom, basis='sto-3g')
 
print("\n Initial FCIcomp Stuff")
print("===========================")
ref = mol.hf_reference

nel = sum(ref)
sz = 0
norb = int(len(ref) / 2)

print(f" nqbit:     {norb*2}")
print(f" nel:       {nel}")
 
fci_comp = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fci_comp.hartree_fock()
 
fock_comp = qf.Computer(norb * 2)

Uhf = qf.utils.state_prep.build_Uprep(ref, 'occupation_list')
fock_comp.apply_circuit(Uhf)
 
# print(fci_comp.str(print_data=True))
 
dim = 2*norb
max_nbody = 2
 
Top = qf.TensorOperator(
    max_nbody = max_nbody,
    dim = dim
    )
 
print("\n SQOP Stuff")
print("===========================")
sq0, sq1, sq2 = mol.sq_hamiltonian.split_by_rank(False)
sqop = qf.SQOperator()
sqop.add_op(sq1)
sqop.add_op(sq2)
 
t_ap12bdy_fock = time.perf_counter()
fock_comp.apply_sq_operator(sqop)
t_ap12bdy_fock = time.perf_counter() - t_ap12bdy_fock

 
# so all seems to be working except
print("\n Tensor Stuff")
print("===========================")
Top.add_sqop_of_rank(sq1, 2)
Top.add_sqop_of_rank(sq2, 4)
 
[H0, H1, H2] = Top.tensors()
 
# print(H1)
 
t_ap12bdy_fci = time.perf_counter()
fci_comp.apply_tensor_spin_12bdy(H1, H2, norb)
t_ap12bdy_fci = time.perf_counter() - t_ap12bdy_fci
 
if(norb < 6): 
    print("\n Final FCIcomp Stuff")
    print("===========================")
    print(fci_comp)

    print("\n Final FockComp Stuff")
    print("===========================")
    print(fock_comp)
 
print("\n Timing")
print("======================================================")
print(f" nqbit:     {norb*2}")
print(f" fci_time:  {t_ap12bdy_fci:12.8f}")
print(f" fock_time: {t_ap12bdy_fock:12.8f}")
print(f" speedup:   {(t_ap12bdy_fock/t_ap12bdy_fci):12.8f}")