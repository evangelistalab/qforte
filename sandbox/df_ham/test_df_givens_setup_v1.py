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

# now get all the df stuff
dfh = mol.df_ham

# Load from the .npz file 
# loaded_data2 = np.load('h4_icut_0.9_dt_0.1_df_tensors.npz')

# eigenvalues = loaded_data2['eigenvalues']
# one_body_squares = loaded_data2['one_body_squares']
# one_body_correction = loaded_data2['one_body_correction']
# scaled_density_density_matrices = loaded_data2['scaled_density_density_matrices']
# basis_change_matrices = loaded_data2['basis_change_matrices']
# time_scaled_bcms = loaded_data2['time_scaled_bcms']
# time_scaled_rr = loaded_data2['time_scaled_rr'] # probably not needed...

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

(
    i_vec,
    j_vec,
    t_vec,
    p_vec,
    diag
) = dfh.givens_decomposition_square(tcms[0], False)

print('\n')
for l in range(len(i_vec)):
    print(f" {i_vec[l]}  {j_vec[l]}  {t_vec[l]}   {p_vec[l]}")

print('\n')

for elem in diag:
    print(f"{elem}")

# from FQE
#  rotations
# (2, 3, 1.5707963267948966, 3.141592653589793)
# (1, 2, 1.0622478791990662, -0.07839331313873395)
# (0, 1, 1.5707963267948966, 3.141592653589793) 
# (2, 3, 1.5707963267948966, 3.141592653589793) 
# (1, 2, 0.6697709944129672, -0.04991284766233473) 

# qf has a 3 2 number that looks wrong...

#   ==> diagonal <== 

#  0.974874+0.222755j
# -0.971242-0.238093j
#  0.979197+0.202911j
#  0.982125+0.188230j 