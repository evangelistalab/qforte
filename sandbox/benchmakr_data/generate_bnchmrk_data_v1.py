import qforte as qf
import numpy as np
 
import time

geom = [
    ('H', (0., 0., 1.0)), 
    ('H', (0., 0., 2.0)),
    ('H', (0., 0., 3.0)), 
    ('H', (0., 0., 4.0)),
    ('H', (0., 0., 5.0)), 
    ('H', (0., 0., 6.0)),
    ('H', (0., 0., 7.0)), 
    ('H', (0., 0., 8.0)),
    ('H', (0., 0., 9.0)), 
    ('H', (0., 0.,10.0))
    ]

mol = qf.system_factory(build_type='psi4', mol_geometry=geom, basis='sto-3g', run_fci=1)
 
print("\n Initial FCIcomp Stuff")
print("===========================")
ref = mol.hf_reference

nel = sum(ref)
sz = 0
norb = int(len(ref) / 2)

# for term in mol.sq_hamiltonian.terms():
#     print(term)

# Your list of tuples
data = mol.sq_hamiltonian.terms()

# Convert the list of tuples to a NumPy array
array_data = np.array(data, dtype=object)

# Save the array to a ZIP file
np.savez('h10_sq_ham', data=array_data)

# Load the data from the ZIP file
# loaded_data = np.load('h4_sq_ham.npz', allow_pickle=True)['data']

# # Iterate through the elements and print them
# for element in loaded_data:
#     print(f"{element[0]} {element[1]} {element[2]}")


