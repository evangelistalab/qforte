import qforte as qf
 
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

# Get the molecule object that now contains both the fermionic and qubit Hamiltonians.
mol = qf.system_factory(build_type='psi4', mol_geometry=geom, basis='sto-3g')

sq_ham = mol.sq_hamiltonian
 
print("\n Initial FCIcomp Stuff")
print("===========================")
ref = mol.hf_reference

nel = sum(ref)
sz = 0
norb = int(len(ref) / 2)

print(f" nqbit:     {norb*2}")
print(f" nel:       {nel}")
 
fci_comp1 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fci_comp1.hartree_fock()

fci_comp2 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fci_comp2.hartree_fock()

temp_pool = qf.SQOpPool()




temp_sqop = qf.SQOperator()

# L = 5

for i in range(len(sq_ham.terms())):
# for i in range(L):
    temp_sqop.add_term(
        sq_ham.terms()[i][0],
        sq_ham.terms()[i][1],
        sq_ham.terms()[i][2]
        )
    
temp_pool.add_hermitian_pairs(1.0, temp_sqop)

# print("\n Temp Sqop")
# print("===========================")
# print(temp_sqop)

# print("\n SQ Hamiltonain")
# print("===========================")
# print(sq_ham)

fci_comp1.apply_sqop(temp_sqop)
print(fci_comp1)

# print("\n SQ Hamiltonain from Hermitian pairs")
# print("===========================")
# print(temp_pool)

fci_comp2.apply_sqop_pool(temp_pool)
print(fci_comp2)