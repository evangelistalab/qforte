import qforte as qf
import numpy as np
from qforte.helper.df_ham_helper import *


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

# now get all the df stuff

dfh = mol.df_ham

# print(dfh.get_ff_eigenvalues())
# print(dfh.get_one_body_squares())
# print(dfh.get_one_body_correction())

sdms = dfh.get_scaled_density_density_matrices()
bcms = dfh.get_basis_change_matrices()
tcms = dfh.get_trotter_basis_change_matrices()

time_scale_first_leaf(dfh, 0.1)

# for i, sdm in enumerate(sdms):
#     print(f"leaf: {i}")
#     print(sdm)

# for i, bcm in enumerate(bcms):
#     print(f"leaf: {i}")
#     print(bcm)

for i, tcm in enumerate(tcms):
    print(f"leaf: {i}")
    print(tcm)

# NOTE, as of 7/9/24 all printed outputs match fqe for H4 assiming 
# the same h1e and h2e tensors are used by both programs.







