import qforte as qf
import numpy as np
from qforte.helper.df_ham_helper import *
import qf_fqe_helper as qfqe

# NOTE(Nick): this sandbox file compares the evolution of the HF wfn
# under trotterized evoltuion of the double factorized hamiltonain  
# to the exact time evolution (via taylor expansion of e^-itH)


def t_diff(Tqf, npt, name, print_both=False):
    print(f"\n  ===> {name} Tensor diff <=== ")
    Tnp = qf.Tensor(shape=np.shape(npt), name='Tnp')
    Tnp.fill_from_nparray(npt.ravel(), np.shape(npt))
    if(print_both):
        print(Tqf)
        print(Tnp)
    Tnp.subtract(Tqf)
    print(f"  ||dT||: {Tnp.norm()}")
    # if(Tnp.norm() > 1.0e-12):
    #     print(Tnp)

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


# geom = [
#     ('H', (0., 0., 1.0)), 
#     ('Be', (0., 0., 2.0)),
#     ('H', (0., 0., 3.0)), 
#     ]

# geom = [('Li', [0.0, 0.0, 0.0]), ('H', [0.0, 0.0, 1.45])]


timer = qf.local_timer()

timer.reset()
# Get the molecule object that now contains both the fermionic and qubit Hamiltonians.
mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g', 
    build_qb_ham = False, ## VERY STRANGE BEHAVIOR!!
    run_fci=1,
    store_mo_ints=1,
    build_df_ham=1,
    df_icut=1.0e-1)

timer.record('Run Psi4 and Initialize')


## ====> Set up Time Step and number of steps <==== ##
dt = 0.2
N = 1
r = 1
order = 1

## ====> Set up DF and Trotter Stuff <==== ##
time_scale_first_leaf(mol.df_ham, dt)
v_lst = mol.df_ham.get_scaled_density_density_matrices()
g_lst = mol.df_ham.get_trotter_basis_change_matrices()

print(f"\nnorb {len(geom)} len v_lst {len(v_lst)} len g_lst {len(g_lst)}\n")

sqham = mol.sq_hamiltonian
hermitian_pairs = qf.SQOpPool()
hermitian_pairs.add_hermitian_pairs(1.0, sqham)

# print(f"sqham: {sqham}")
# print(f"hermitian pairs: {hermitian_pairs}")

print("")
print(f"len(sqham.terms()):  {len(sqham.terms())} ")
print(f"len(hermitian_pairs.terms()):  {len(hermitian_pairs.terms())} ")
print("")


## ====> set up FCIComputers <==== ##
ref = mol.hf_reference

nel = sum(ref)
sz = 0
norb = int(len(ref) / 2)

print(f"nel {nel} norb {norb}")

fc1 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fc2 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fc3 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)

fc1.hartree_fock()
fc2.hartree_fock()
fc3.hartree_fock()


# =====> Print FQE OPS <=====
# print("\n\n#==> Begin Op String <==\n")
# my_sqham_str = qfqe.get_fqe_op_string(sqham)
# # print(my_sqham_str)
# print("\n\n#==> End Op String Formation <== \n\n")

# print("\n\n#==> Begin HP Op String <==\n")
# my_hp_str = qfqe.get_fqe_hp_string(hermitian_pairs)
# # print(my_hp_str)
# print("\n\n#==> End HP Op String Formation <== \n\n")


# Convert the string to a NumPy array
# string_array = np.array([my_hp_str], dtype='str')

# Define the file path
# file_path = "my_hp_str.txt"

# # Save the NumPy array to a file
# np.savetxt(file_path, string_array, fmt='%s')

## ====> Load from FQE <==== ##
loaded_data = np.load('wfn_evo_individ_ops_H4_v1.npz')
frm_hf_individ_ops = loaded_data['frm_hf_individ_ops']
frm_hf_cumulative_ops = loaded_data['frm_hf_cumulative_ops']
wfn_ary_all = loaded_data['wfn_ary_all']
wfn_ary_exact = loaded_data['wfn_ary_exact']



## ====> DF Evolution <==== ##
print(f"dt:    {dt}")
print(f"r:     {r}")
print(f"order: {order}")

print("\n\n")

# print(sqham)

print(f"\nmol.nuclear_repulsion_energy: {mol.nuclear_repulsion_energy}")
print(f"\nexp(-i*dt*nre): {np.exp(-1.0j*dt*mol.nuclear_repulsion_energy)}")
print("")

gphase = np.exp(-1.0j*dt*mol.nuclear_repulsion_energy)


fc1.evolve_op_taylor(
    sqham,
    dt,
    1.0e-15,
    30)

fc2.evolve_pool_trotter(
    hermitian_pairs,
    dt,
    r,
    order,
    antiherm=False,
    adjoint=False)


for i, hp in enumerate(hermitian_pairs.terms()):
    coeff = hp[0]
    sqop = hp[1]

    fc3.apply_sqop_evolution(
        dt, 
        sqop)
    
    t_diff(
        fc3.get_state_deep(), 
        frm_hf_cumulative_ops[i], 
        f"|| Cqf_{i} - Cfqe_{i} ||", 
        print_both=False)


# print(fc1.str(print_complex=True))
# print(fc2.str(print_complex=True))
# print(fc3.str(print_complex=True))

# print(fc1.get_state().norm())
# print(fc2.get_state().norm())
# print(fc3.get_state().norm())

E1 = np.real(fc1.get_exp_val(sqham))
E2 = np.real(fc2.get_exp_val(sqham))
E3 = np.real(fc3.get_exp_val(sqham))

C1 = fc1.get_state_deep()
dC2 = fc2.get_state_deep()
dC3 = fc3.get_state_deep()

dC2.subtract(C1)
dC3.subtract(C1)

print(f"t {dt:6.6f} |dC2| {dC2.norm():6.6f} |dC3| {dC3.norm():6.6f}  {E1:6.6f} {E2:6.6f} {E3:6.6f}")

# print("")
# print(hermitian_pairs.terms()[15])
# print("")
# print(hermitian_pairs.terms()[36])
# print("")









