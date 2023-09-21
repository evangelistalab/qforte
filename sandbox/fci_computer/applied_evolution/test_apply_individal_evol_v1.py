import qforte as qf
import numpy as np

nel = 4
sz = 0
norb = 4

fci_comp = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fci_comp.hartree_fock()

random_array = np.random.rand(fci_comp.get_state().shape()[0], fci_comp.get_state().shape()[1])
random = np.array(random_array, dtype = np.dtype(np.complex128))

random = np.ones((fci_comp.get_state().shape()[0], fci_comp.get_state().shape()[1]))

Crand = qf.Tensor(fci_comp.get_state().shape(), "Crand")
Crand.fill_from_nparray(random.ravel(), Crand.shape())
rand_nrm = Crand.norm()
# Crand.scale(1/rand_nrm)

# fci_comp.set_state(Crand)

# print(fci_comp.str(print_data=True))

dim = 2*norb
max_nbody = 1

# This one makes the wfn complex
# op = (FermionOperator('7^ 6^ 3 2', coefficient= theta) + FermionOperator('2^ 3^ 6 7', coefficient = theta))
# time = 1.0

# This one keeps it real
# op = (FermionOperator('7^ 6^ 3 2', coefficient=-1j * theta) + FermionOperator('2^ 3^ 6 7', coefficient=1j * theta))
# time = 1.0

print("\n SQOP Stuff")
print("===========================")
sqop = qf.SQOperator()

# should make the wfn complex
# sqop.add_term( +0.704645, [7, 6], [3, 2])
# sqop.add_term( +0.704645, [2, 3], [6, 7])

# should make the wfn real (there is a sign flip somewhere)...
sqop.add_term( +0.704645 * 1.0j, [7, 6], [3, 2])
sqop.add_term( -0.704645 * 1.0j, [2, 3], [6, 7])
time = 1.0

print(sqop)


print("\n Initial FCIcomp Stuff")
print("===========================")
print(fci_comp)

fci_comp.apply_sqop_evolution(time, sqop)

print("\n Final FCIcomp Stuff")
print("===========================")
print(fci_comp.str(print_data=True, print_complex=True))