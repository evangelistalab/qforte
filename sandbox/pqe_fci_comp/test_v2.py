import qforte as qf
r = 0.7

geom = [
    ('H', (0., 0., 0.0*r)), 
    ('H', (0., 0., 1.0*r)),
    ('H', (0., 0., 2.0*r)),
    ('H', (0., 0., 3.0*r)),
    # ('H', (0., 0., 4.0*r)), 
    # ('H', (0., 0., 5.0*r)),
    # ('H', (0., 0., 6.0*r)),
    # ('H', (0., 0., 7.0*r)),
    # ('H', (0., 0., 8.0*r)),
    # ('H', (0., 0., 9.0*r))
    ]

timer = qf.local_timer()

mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g',
    run_fci=1)

psi_4_time = timer.get()

# alg_fock = qf.UCCNPQE(
#     mol,
#     computer_type = 'fock'
#     )

# alg_fock.run(opt_thresh=1.0e-2, pool_type='SD')
# print(f'\n\n Efci:   {mol.fci_energy:+12.10f}')

timer.reset()
alg_fci = qf.UCCNPQE(
    mol,
    computer_type = 'fci',
    verbose=True)
setup_fci_time = timer.get()

timer.reset()
alg_fci.run(opt_thresh=1.0e-2, pool_type='SD')
run_fci_time = timer.get()

print(f' Efci:    {mol.fci_energy:+12.10f}')
print(f' psi_4_time:      {psi_4_time}')
print(f' setup_fci_time:  {setup_fci_time}')
print(f' run_fci_time:    {run_fci_time}')

#  Time1:   1.955370494
