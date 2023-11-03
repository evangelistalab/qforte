import qforte as qf


geom = [
    ('H', (0., 0., 1.00)), 
    ('H', (0., 0., 2.00)),
    ('H', (0., 0., 3.00)),
    ('H', (0., 0., 4.00)),
    ('H', (0., 0., 5.00)), 
    ('H', (0., 0., 6.00)),
    ('H', (0., 0., 7.00)),
    ('H', (0., 0., 8.00)),
    ('H', (0., 0., 9.00)),
    ('H', (0., 0., 10.00))
    ]

mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g',
    run_fci=1)


s = 8
dt = 0.2

# alg_fock = qf.SRQK(
#     mol,
#     computer_type = 'fock',
#     trotter_number=4
#     )

# alg_fock.run(
#     s=s,
#     dt=dt
# )

# print(f'\n\n Efci:   {mol.fci_energy:+12.10f}')


alg_fci = qf.SRQK(
    mol,
    computer_type = 'fci',
    trotter_number=10
    )

alg_fci.run(
    s=s,
    dt=dt
    )

Egs = alg_fci.get_gs_energy()

print('\n\n')
print(f' Efci:   {mol.fci_energy:+12.10f}')
print(f' Egs:    {Egs:+12.10f}')
print(f' dE:     {Egs-mol.fci_energy:+12.10f}')
