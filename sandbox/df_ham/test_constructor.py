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



timer.reset()

# Load h1e from the .npz file
loaded_data = np.load('mol_e0_h1e_h2e.npz')
e0 = loaded_data['e0']
h1e = loaded_data['h1e']
h2e = loaded_data['h2e']
timer.record('Run Psi4 and Initialize')

print("\nLoaded h1e:\n")
print(h1e)

n=3
diagonal = h1e[range(n), range(n)]

print(f"\ndiagonal: {diagonal}")

# LGTM