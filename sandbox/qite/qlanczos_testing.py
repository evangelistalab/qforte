# import numpy as np
# # n_dim = 11

# # for i in range(0, n_dim):
# #     i_2 = 2*i
# #     for j in range(i+1):
# #         j_2 = 2*j
# #         r = (i_2 + j_2) // 2
# #         print(r, end=' ')
# #     print('\n')

# c_list = [1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# c_dim = 11
# n_dim = (c_dim+1)//2+1

# print(f'c_dim:{c_dim}')
# print(f'n_dim:{n_dim}')

# h_mat = np.zeros((n_dim, n_dim))
# s_mat = np.zeros((n_dim, n_dim))

# print(s_mat)
# # print(h_mat)
# print('\n')

# # build elements of S and H matrix
# # only works with lanczos gap of 2
# for i in range(n_dim):
#     i_2 = 2*i
#     for j in range(i+1):
#         j_2 = 2*j
#         r = (i_2 + j_2) // 2
        
#         n = 1
#         d = 1

#         print(f'loop1: {range(j_2+1, r+1)}')
#         for ix in range(j_2+1, r+1):
#             n *= c_list[ix]

#         print(f'loop2: {range(r+1, i_2+1)}')
#         for ix in range(r+1, i_2+1):
#             d *= c_list[ix]

#         s_mat[j,i] = s_mat[i,j] = round(np.sqrt(n / d),1)
#         # h_mat[j,i] = h_mat[i,j] = s_mat[i,j] * 1.0 + -1.0j
    
#     k = i+1
#     # print(k)
#     # print(s_mat[0:k, 0:k])
#     print(s_mat)
#     print('\n')
#     # print(h_mat)

from qforte import Molecule, QLANCZOS, system_factory

# The FCI energy for H2 at 1.5 Angstrom in a sto-3g basis
# E_fci = -0.9981493534
# E_fock = -0.9108735544

print('\nBuild Psi4 Geometry')
print('-------------------------')

geom = [('H', (0., 0., 0.)), 
        ('H', (0., 0., .50)),
        ('H', (0., 0., 1.00)), 
        ('H', (0., 0., 1.50)),
        ('H', (0., 0., 2.00)), 
        ('H', (0., 0., 2.50)),
        ('H', (0., 0., 3.00)), 
        ('H', (0., 0., 3.50))]
        # ('H', (0., 0., 4.00)), 
        # ('H', (0., 0., 4.50))]

# Get the molecule object that now contains both the fermionic and qubit Hamiltonians.
mol = system_factory(build_type='psi4', mol_geometry=geom, basis='sto-6g', run_fci=True, run_ccsd=False)
# mol.ccsd_energy = 0.0

print(f'The FCI energy from Psi4:                                    {mol.fci_energy:12.10f}')
print(f'The HF energy from Psi4:                                     {mol.hf_energy:12.10f}')


print('\nBegin QLanzcos test for H4 with exact (unphysical) matrix construction')
print('-------------------------')

alg = QLANCZOS(mol, computer_type='fci')
alg.run(beta=10, db=0.5, second_order=True, lanczos_gap=4, realistic_lanczos=False)
# Egs_FCI_low_mem = alg.get_gs_energy()

print(f'The HF energy from Psi4:                                     {mol.hf_energy:12.10f}')
print(f'The FCI energy from Psi4:                                    {mol.fci_energy:12.10f}')
# print(f'The FCI energy from QLanczos:                                {Egs_FCI:12.10f}')