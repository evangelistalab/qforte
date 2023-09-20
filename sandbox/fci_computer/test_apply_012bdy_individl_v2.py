import qforte as qf
import numpy as np

import time

# Define the reference and geometry lists.
geom = [
    ('H', (0., 0., 1.0)), 
    ('H', (0., 0., 2.0)),
    ('H', (0., 0., 3.0)), 
    ('H', (0., 0., 4.0)),
    # ('H', (0., 0., 5.0)), 
    # ('H', (0., 0., 6.0)),
    # ('H', (0., 0., 7.0)), 
    # ('H', (0., 0., 8.0)),
    # ('H', (0., 0., 9.0)), 
    # ('H', (0., 0.,10.0))
    ]

reference = 'random'
# reference = 'hf'

# Get the molecule object that now contains both the fermionic and qubit Hamiltonians.
mol = qf.system_factory(build_type='psi4', mol_geometry=geom, basis='sto-3g')

mol.sq_hamiltonian.canonical_order()
 
print("\n Initial FCIcomp Stuff")
print("===========================")
ref = mol.hf_reference

nel = sum(ref)
sz = 0
norb = int(len(ref) / 2)

print(f" nqbit:     {norb*2}")
print(f" nel:       {nel}")
 
fci_comp = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fci_comp2 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)

# For Random Stae
random_array = np.random.rand(fci_comp.get_state().shape()[0], fci_comp.get_state().shape()[1])
random = np.array(random_array, dtype = np.dtype(np.complex128))

Crand = qf.Tensor(fci_comp.get_state().shape(), "Crand")

Crand.fill_from_nparray(random.ravel(), Crand.shape())
rand_nrm = Crand.norm()
Crand.scale(1/rand_nrm)

print(Crand)

# fock_comp = qf.Computer(norb * 2)
# Uhf = qf.utils.state_prep.build_Uprep(ref, 'occupation_list')
# fock_comp.apply_circuit(Uhf)
#  qb_ham = mol.sq_hamiltonian.jw_transform()
 
dim = 2*norb

bad_sqops = []

bad_Crefs = []
bad_Cnews = []
bad_Cdif_norms = []

for sqop_term in mol.sq_hamiltonian.terms():
    term_rank = int(2 * len(sqop_term[1]))
    print(f"rank: {term_rank}")
    sqop = qf.SQOperator()
    sqop.add_term(sqop_term[0], sqop_term[1], sqop_term[2])

    max_nbody = int(term_rank / 2)

    Top = qf.TensorOperator(
        max_nbody = max_nbody,
        dim = dim
        )

    Top.add_sqop_of_rank(sqop, term_rank)

    if(reference == 'hf'):
        fci_comp.hartree_fock()
        fci_comp2.hartree_fock()

    elif(reference == 'random'):
        fci_comp.set_state(Crand)
        fci_comp2.set_state(Crand)


    if term_rank == 2:
        [H0, H1] = Top.tensors()
        fci_comp.apply_tensor_spin_1bdy(H1, norb)
        fci_comp2.apply_sqop(sqop)

    if term_rank == 4:
        [H0, H1, H2] = Top.tensors()
        fci_comp.apply_tensor_spin_012bdy(H0, H1, H2, norb)
        fci_comp2.apply_sqop(sqop)


    Cref = fci_comp.get_state_deep()
    Cnew = fci_comp2.get_state_deep()
    Cdif = fci_comp2.get_state_deep()

    Cdif.subtract(Cref)
    norm = Cdif.norm()

    if(norm > 1.0e-6): 

        bad_sqops.append(sqop)
        bad_Crefs.append(Cref)
        bad_Cnews.append(Cnew)
        bad_Cdif_norms.append(norm)

        
for i, bad_sqop in enumerate(bad_sqops):
    print("\n\n  Bad SQOP")
    print("  ======================================================")
    print(bad_sqops[i])
    print(f"  || Cdif ||: {bad_Cdif_norms[i]}")

    print("\n  Final FCIcomp Stuff")
    print("  ===========================")
    print(bad_Crefs[i])

    print("\n  Final FCI2comp Stuff")
    print("  ===========================")
    print(bad_Cnews[i])

print("\n  Summary")
print("  ===========================")
print(len(mol.sq_hamiltonian.terms()))
print(len(bad_Cdif_norms))

        



# max_nbody = 2

 
# Top = qf.TensorOperator(
#     max_nbody = max_nbody,
#     dim = dim
#     )
 
# print("\n SQOP Stuff")
# print("===========================")
# sq0, sq1, sq2 = mol.sq_hamiltonian.split_by_rank(False)
# sqop = qf.SQOperator()
# sqop.add_op(sq1)
# sqop.add_op(sq2)
 
# t_ap012bdy_fock = time.perf_counter()
# Eo_fock = fock_comp.direct_op_exp_val(qb_ham)
# t_ap012bdy_fock = time.perf_counter() - t_ap012bdy_fock

 
# so all seems to be working except
# print("\n Tensor Stuff")
# print("===========================")
# Top.add_sqop_of_rank(sq0, 0)
# Top.add_sqop_of_rank(sq1, 2)
# Top.add_sqop_of_rank(sq2, 4)
 
# [H0, H1, H2] = Top.tensors()
 
# # print(H1)
 
# t_ap012bdy_fci = time.perf_counter()
# fci_comp.apply_tensor_spin_012bdy(H0, H1, H2, norb)
# Eo_fci = fci_comp.get_hf_dot() 
# t_ap012bdy_fci = time.perf_counter() - t_ap012bdy_fci


# t_ap012bdy_fci2 = time.perf_counter()
# fci_comp2.apply_sqop(mol.sq_hamiltonian)
# Eo_fci2 = fci_comp2.get_hf_dot() 
# t_ap012bdy_fci2 = time.perf_counter() - t_ap012bdy_fci2
 
# if(norb < 6): 
#     print("\n Final FCIcomp Stuff")
#     print("===========================")
#     print(fci_comp)

#     print("\n Final FCI2comp Stuff")
#     print("===========================")
#     print(fci_comp2)

#     print("\n Final FockComp Stuff")
#     print("===========================")
#     print(fock_comp)
 
# print("\n Timing")
# print("======================================================")
# print(f" nqbit:     {norb*2}")
# print(f" fci_time:  {t_ap012bdy_fci:12.8f}")
# print(f" fci_time2: {t_ap012bdy_fci2:12.8f}")
# print(f" fock_time: {t_ap012bdy_fock:12.8f}")
# print(f" speedup:   {(t_ap012bdy_fock/t_ap012bdy_fci):12.8f}")
# print(f" Eo fock:   {Eo_fock}")
# print(f" Eo fci:    {Eo_fci}")
# print(f" Eo fci2:    {Eo_fci2}")