import qforte as qf
import numpy as np
from qforte.helper.df_ham_helper import *

# NOTE(Nick): this sandbox file compares the evolution of the HF wfn
# under trotterized evoltuion of only the first double factorized 't-leaf' 
# relative to fqe, as of 7/12/2024 evolution matches.


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


# Define the reference and geometry lists.
geom = [
    ('H', (0., 0., 1.0)), 
    ('H', (0., 0., 2.0)),
    ('H', (0., 0., 3.0)), 
    ('H', (0., 0., 4.0)),
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

dfh = mol.df_ham

# Load fqe data from the .npz file
loaded_data = np.load('h4_icut_0.9_dt_0.1_df_evo_wfns.npz')
flga_ary = loaded_data['flga_ary']
flgb_ary = loaded_data['flgb_ary']
flg_ary = loaded_data['flg_ary']
fld_ary = loaded_data['fld_ary']
full_df_evo_ary = loaded_data['full_df_evo_ary']


## ====> Set up DF Trotter Stuff <==== ##
dt = 0.1
time_scale_first_leaf(dfh, dt)

v_lst = dfh.get_scaled_density_density_matrices()
g_lst = dfh.get_trotter_basis_change_matrices()

## ====> set up FCIComputer <==== ##

nel = 4
sz = 0
norb = 4

fci_comp = qf.FCIComputer(nel=nel, sz=sz, norb=norb)


## ====> First leaf alpha givens <==== ##
fci_comp.hartree_fock()
fci_comp.evolve_givens(
    g_lst[0],
    True # is alpha
)

t_diff(fci_comp.get_state_deep(), flga_ary, "flga_ary", print_both=False)


## ====> First leaf beta givens <==== ##
fci_comp.hartree_fock()
fci_comp.evolve_givens(
    g_lst[0],
    False # is alpha
)
t_diff(fci_comp.get_state_deep(), flgb_ary, "flgb_ary", print_both=False)

## ====> First leaf both givens <==== ##
fci_comp.hartree_fock()
fci_comp.evolve_givens(
    g_lst[0],
    True # is alpha
)

fci_comp.evolve_givens(
    g_lst[0],
    False # is alpha
)
t_diff(fci_comp.get_state_deep(), flg_ary, "flg_ary", print_both=False)


## ====> First leaf diagonal <==== ##
fci_comp.hartree_fock()
fci_comp.evolve_diagonal_from_mat(
            v_lst[0],
            dt
        );
t_diff(fci_comp.get_state_deep(), fld_ary, "fld_ary", print_both=False)


## ====> Whole Enchiladda <==== ##
fci_comp.hartree_fock()
fci_comp.evolve_df_ham_trotter(
      dfh,
      dt)
t_diff(fci_comp.get_state_deep(), full_df_evo_ary, "full_df_evo_ary", print_both=False)










