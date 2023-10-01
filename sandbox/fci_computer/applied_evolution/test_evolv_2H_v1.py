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
 
fci_comp = qf.FCIComputer(nel=nel, sz=sz, norb=norb)


# reference = 'random'
reference = 'hf'

if(reference == 'hf'):
    fci_comp.hartree_fock()
    

elif(reference == 'random'):
    np.random.seed(42)
    random_array = np.random.rand(fci_comp.get_state().shape()[0], fci_comp.get_state().shape()[1])
    random = np.array(random_array, dtype = np.dtype(np.complex128))

    Crand = qf.Tensor(fci_comp.get_state().shape(), "Crand")
    Crand.fill_from_nparray(random.ravel(), Crand.shape())
    rand_nrm = Crand.norm()
    Crand.scale(1/rand_nrm)

    fci_comp.set_state(Crand)
 
dim = 2*norb
max_nbody = 2

t_app_trotter = time.perf_counter()

counter = 0

for term in mol.sq_hamiltonian.terms():

    if counter == 0:
        counter += 1
        continue

    term_sqop = qf.SQOperator()
    term_sqop.add_term(term[0], term[1], term[2])
    term_sqop.add_term(term[0], term[2][::-1], term[1][::-1])

    print(term_sqop)

    fci_comp.apply_sqop_evolution(0.1, term_sqop)

    if counter < 4:
        print(fci_comp)
    
    counter += 1

t_app_trotter = time.perf_counter() - t_app_trotter

Eo_fci = fci_comp.get_hf_dot() 
 
if(norb < 6): 
    print("\n Final FCIcomp Stuff")
    print("===========================")
    print(fci_comp)
 
print("\n Timing")
print("======================================================")
print(f" nqbit:     {norb*2}")
print(f" fci_time:  {t_app_trotter:12.8f}")


print(f" Efci:      {mol.fci_energy}")
print(f" Ehf:       {mol.hf_energy}")
print(f" Eo fci:    {Eo_fci}")
