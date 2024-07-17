import qforte as qf


geom = [
    # ('Be', (0., 0., 1.00)), 
    ('H', (0., 0., 1.00)),
    ('H', (0., 0., 2.00)),
    ('H', (0., 0., 3.00)),
    ('H', (0., 0., 4.00)),
    # ('H', (0., 0., 5.00)),
    # ('H', (0., 0., 6.00)),
    # ('H', (0., 0., 7.00)),
    # ('H', (0., 0., 8.00)),
    # ('H', (0., 0., 9.00)),
    # ('H', (0., 0., 10.00))
    ]

timer = qf.local_timer()

timer.reset()

mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g',
    run_fci=1)

timer.record("Psi4 Setup")



alg_fock = qf.UCCNVQE(
    mol,
    computer_type = 'fock'
    )


timer.reset()

alg_fock.run(
    opt_thresh=1.0e-4, 
    pool_type='SD',
    optimizer='BFGS'
    )

timer.record("dUCC Fock")

print(f'\n\n Efci:   {mol.fci_energy:+12.10f}')


alg_fci = qf.UCCNVQE(
    mol,
    computer_type = 'fci',
    )

timer.reset()
alg_fci.run(opt_thresh=1.0e-4, 
            pool_type='SD',
            optimizer='BFGS',
            # apply_ham_as_tensor = True
            )

timer.record("dUCC FCI")

print(timer)
            
print(f'\n\n Efci:   {mol.fci_energy:+12.10f}')

alg_fci = qf.UCCNVQE(
    mol,
    computer_type = 'fci',
    )

timer.reset()
alg_fci.run(opt_thresh=1.0e-4, 
            pool_type='SD',
            optimizer='BFGS',
            # apply_ham_as_tensor = False
            )

timer.record("dUCC FCI")

print(timer)
            
print(f'\n\n Efci:   {mol.fci_energy:+12.10f}')