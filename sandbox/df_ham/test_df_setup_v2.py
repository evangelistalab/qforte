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

# ffes = dfh.get_ff_eigenvalues()
# obss = dfh.get_one_body_squares()
# obcs = dfh.get_one_body_correction()
# sdms = dfh.get_scaled_density_density_matrices()
# bcms = dfh.get_basis_change_matrices()
# tcms = dfh.get_trotter_basis_change_matrices()

# time_scale_first_leaf(dfh, 0.1)

# for i, sdm in enumerate(sdms):
#     print(f"leaf: {i}")
#     print(sdm)

# for i, bcm in enumerate(bcms):
#     print(f"leaf: {i}")
#     print(bcm)

# for i, tcm in enumerate(tcms):
#     print(f"leaf: {i}")
#     print(tcm)

# Load from the .npz file 
loaded_data2 = np.load('h4_icut_0.9_dt_0.1_df_tensors.npz')

eigenvalues = loaded_data2['eigenvalues']
one_body_squares = loaded_data2['one_body_squares']
one_body_correction = loaded_data2['one_body_correction']
scaled_density_density_matrices = loaded_data2['scaled_density_density_matrices']
basis_change_matrices = loaded_data2['basis_change_matrices']
time_scaled_bcms = loaded_data2['time_scaled_bcms']
time_scaled_rr = loaded_data2['time_scaled_rr'] # probably not needed...

# print(loaded_data2)

ffes = dfh.get_ff_eigenvalues()
obss = dfh.get_one_body_squares()
obcs = dfh.get_one_body_correction()
sdms = dfh.get_scaled_density_density_matrices()
bcms = dfh.get_basis_change_matrices()

time_scale_first_leaf(dfh, 0.1)
tcms = dfh.get_trotter_basis_change_matrices()


def t_diff(Tqf, npt, name, print_both=False):
    print(f"\n  ===> {name} Tensor diff <=== ")
    Tnp = qf.Tensor(shape=np.shape(npt), name='Tnp')
    Tnp.fill_from_nparray(npt.ravel(), np.shape(npt))
    if(print_both):
        print(Tqf)
        print(Tnp)
    Tnp.subtract(Tqf)
    print(f"  ||dT||: {Tnp.norm()}")
    if(Tnp.norm() > 1.0e-12):
        print(Tnp)

def tl_diff(qftl, nptl, name, print_both=False):
    if(len(qftl) != len(nptl)):
        raise ValueError("Not the same number of Leaves!")
    for l in range(len(qftl)):
        t_diff(qftl[l], nptl[l], f"{name} leaf: {l}", print_both=print_both)


print(type(eigenvalues))
# print(type(scaled_density_density_matrices))

# ====> compare ffes < <====
t_diff(ffes, eigenvalues, "eigenvalues")

# ====> compare obss < <====
t_diff(obss, one_body_squares, "one_body_squares")

# ====> compare obcs < <====
t_diff(obcs, one_body_correction, "one_body_correction", print_both=False)

# ====> compare sdms < <====
tl_diff(sdms, scaled_density_density_matrices, "scaled_density_density_matrices")

# ====> compare bcms < <====
tl_diff(bcms, basis_change_matrices, "basis_change_matrices")

# ====> compare tcms < <====
tl_diff(tcms, time_scaled_bcms, "time_scaled_bcms", print_both=False)



