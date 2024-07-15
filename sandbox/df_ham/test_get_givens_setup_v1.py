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
    # ('H', (0., 0.,10.0)),
    # ('H', (0., 0.,11.0)), 
    # ('H', (0., 0.,12.0))
    ]


timer = qf.local_timer()

timer.reset()
# Get the molecule object that now contains both the fermionic and qubit Hamiltonians.
mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g', 
    build_qb_ham = False,
    run_fci=1,
    store_mo_ints=1,
    build_df_ham=1)

timer.record('Run Psi4 and Initialize')

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
N = 1

print(f"dt:    {time}")
print(f"r:     {r}")
print(f"order: {order}")

for _ in range(N):
    fci_comp1.evolve_pool_trotter(
        hermitian_pairs,
        time,
        r,
        order,
        antiherm=False,
        adjoint=False)

    C1 = fci_comp1.get_state_deep()

    ## ===> Where all the setup of the givens stuff goes <=== ##
    
    fci_comp2.evolve_df_hamiltonain(
        mol.df_ham,
        time)

    
    C2 = fci_comp2.get_state_deep()
    dC = fci_comp2.get_state_deep()



    dC.subtract(C1)

    # print(C1)
    print(f"deltaC.norm() {dC.norm()}")

# not working rn, I suspect evolution is correct but formation of 
# hermitian pairs might be funky for diagonal part of the
# hamiltonain, or some such...



