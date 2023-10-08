import qforte as qf


geom = [
    ('Be', (0., 0., 1.00)), 
    ('H', (0., 0., 2.00)),
    ('H', (0., 0., 0.00)),
    # ('H', (0., 0., 4.00))
    ]

mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g',
    run_fci=1)

alg_fock = qf.UCCNPQE(
    mol,
    computer_type = 'fock'
    )

alg_fock.run(opt_thresh=1.0e-2, pool_type='SDT')
print(f'\n\n Efci:   {mol.fci_energy:+12.10f}')


alg_fci = qf.UCCNPQE(
    mol,
    computer_type = 'fci'
    )

alg_fci.run(opt_thresh=1.0e-2, pool_type='SDT')
print(f'\n\n Efci:   {mol.fci_energy:+12.10f}')
