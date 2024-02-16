import qforte as qf
import numpy as np
 
import time

# Define the reference and geometry lists.
geom = [
    ('H', (0., 0., 1.0)), 
    ('H', (0., 0., 2.0)),
    ('H', (0., 0., 3.0)), 
    ('H', (0., 0., 4.0)),
    ('H', (0., 0., 5.0)), 
    ('H', (0., 0., 6.0)),
    # ('H', (0., 0., 7.0)), 
    # ('H', (0., 0., 8.0)),
    # ('H', (0., 0., 9.0)), 
    # ('H', (0., 0.,10.0))
    ]

# Get the molecule object that now contains both the fermionic and qubit Hamiltonians.
mol = qf.system_factory(build_type='psi4', mol_geometry=geom, basis='sto-3g', run_fci=1)
 
print("\n Initial FCIcomp Stuff")
print("===========================")
ref = mol.hf_reference

nel = sum(ref)
sz = 0
norb = int(len(ref) / 2)

print(f" nqbit:     {norb*2}")
print(f" nel:       {nel}")
 
fci_comp1 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fci_comp2 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)

# reference = 'random'
reference = 'hf'

if(reference == 'hf'):
    fci_comp1.hartree_fock()
    fci_comp2.hartree_fock()
    

# elif(reference == 'random'):
#     np.random.seed(42)
#     random_array = np.random.rand(fci_comp.get_state().shape()[0], fci_comp.get_state().shape()[1])
#     random = np.array(random_array, dtype = np.dtype(np.complex128))

#     Crand = qf.Tensor(fci_comp.get_state().shape(), "Crand")
#     Crand.fill_from_nparray(random.ravel(), Crand.shape())
#     rand_nrm = Crand.norm()
#     Crand.scale(1/rand_nrm)

#     fci_comp.set_state(Crand)
    
sqham = mol.sq_hamiltonian
# sqham.simplify()

hermitian_pairs = qf.SQOpPool()
hermitian_pairs.add_hermitian_pairs(1.0, sqham)


# print('sqham')
# print(sqham)

# print('hemitian_pairs')
# print(hermitian_pairs)

time = 0.1

r = 1
order = 1

print(f"dt:    {time}")
print(f"r:     {r}")
print(f"order: {order}")

for _ in range(10):
# Call Trotter for fci_comp1
    fci_comp1.evolve_pool_trotter(
        hermitian_pairs,
        time,
        r,
        order,
        antiherm=False,
        adjoint=False)


    # print(fci_comp1.str(print_complex=False))
    # print(fci_comp1.get_state().norm())

    # Call full taylor evolution for fci_comp2
    fci_comp2.evolve_op_taylor(
        sqham,
        time,
        1.0e-15,
        30)

    # print(fci_comp2)
    # print(fci_comp2.get_state().norm())

    C1 = fci_comp1.get_state_deep()
    C2 = fci_comp2.get_state_deep()

    C1.subtract(C2)

    # print(C1)
    print(f"deltaC.norm() {C1.norm()}")

# not working rn, I suspect evolution is correct but formation of 
# hermitian pairs might be funky for diagonal part of the
# hamiltonain, or some such...

