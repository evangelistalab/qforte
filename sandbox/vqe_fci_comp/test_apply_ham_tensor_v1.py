import qforte as qf


geom = [
    # ('Be', (0., 0., 1.00)), 
    ('H', (0., 0., 1.00)),
    ('H', (0., 0., 2.00)),
    ('H', (0., 0., 3.00)),
    ('H', (0., 0., 4.00)),
    ('H', (0., 0., 5.00)),
    ('H', (0., 0., 6.00)),
    ('H', (0., 0., 7.00)),
    ('H', (0., 0., 8.00)),
    # ('H', (0., 0., 9.00)),
    # ('H', (0., 0., 10.00))
    ]

mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g',
    run_fci=1)

# alg_fock = qf.UCCNVQE(
#     mol,
#     computer_type = 'fock'
#     )

# alg_fock.run(
#     opt_thresh=1.0e-2, 
#     pool_type='GSD',
#     # optimizer='LBFGS'
#     )

# print(f'\n\n Efci:   {mol.fci_energy:+12.10f}')


apply_ham_as_tensor = True


alg_fci = qf.UCCNVQE(
    mol,
    computer_type = 'fci',
    apply_ham_as_tensor=apply_ham_as_tensor,
    )



timer = qf.local_timer()
timer.reset()


alg_fci.run(opt_thresh=1.0e-3, 
            pool_type='SD',
            optimizer='BFGS')

timer.record(f"Run SRQK FCI")


print(f"\n\nApply ham as tensor: {apply_ham_as_tensor}")
print(timer)

print(f'\n\n Efci:   {mol.fci_energy:+12.10f}')
