import qforte as qf
import numpy as np

nel = 4
sz = 0
norb = 4

fci_comp = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fci_comp.hartree_fock()

fci_comp2 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fci_comp2.hartree_fock()


rand = False
if(rand):
    # random_array = np.random.rand(fci_comp.get_state().shape()[0], fci_comp.get_state().shape()[1])
    # random = np.array(random_array, dtype = np.dtype(np.complex128))
    random = np.ones((fci_comp.get_state().shape()[0], fci_comp.get_state().shape()[1]))
    Crand = qf.Tensor(fci_comp.get_state().shape(), "Crand")
    Crand.fill_from_nparray(random.ravel(), Crand.shape())
    rand_nrm = Crand.norm()
    Crand.scale(1/rand_nrm)
    fci_comp.set_state(Crand)
    print(fci_comp.str(print_data=True))

dim = 2*norb
max_nbody = 1

print("\n Initial FCIcomp Stuff")
print("===========================")
print(fci_comp)


sq_terms = [
    (+0.704645, [7, 6], [3, 2]), # 2body ab 
    (+0.4, [6], [0]), # 1bdy-a
    (+0.4, [7], [3]), # 1bdy-a
    ]

time = 1.0
print_imag = True

pool = qf.SQOpPool()

for sq_term in sq_terms:

    sqop = qf.SQOperator()
    sqop.add_term(sq_term[0], sq_term[1], sq_term[2])
    sqop.add_term(np.conj(sq_term[0]), sq_term[2][::-1], sq_term[1][::-1])

    pool.add_term(1.0, sqop)

    print("\n SQOP Stuff")
    print("===========================")
    print(sqop)
    fci_comp.apply_sqop_evolution(
        time, 
        sqop,
        antiherm=True)

print("\n SQOP Pool Stuff")
print("===========================")
print(pool)

print("\n Final FCIcomp Stuff")
print("===========================")

Ctemp = fci_comp.get_state_deep()
cnrm = Ctemp.norm()
print(f"||C||: {cnrm}")
print(fci_comp.str(print_data=True, print_complex=print_imag))

fci_comp2.evolve_pool_trotter_basic(
    pool,
    antiherm=True)

print("\n Final FCIcomp2 Stuff")
print("===========================")
Ctemp2 = fci_comp2.get_state_deep()
cnrm2 = Ctemp2.norm()
print(f"||C||: {cnrm2}")
print(fci_comp2.str(print_data=True, print_complex=print_imag))
