import qforte as qf


geom = [
    ('H', (0., 0., 1.00)), 
    ('H', (0., 0., 2.00)),
    ('H', (0., 0., 3.00)),
    ('H', (0., 0., 4.00))
    ]

mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g',
    run_fci=1)


s = 4
dt = 0.2

alg_fock = qf.SRQK(
    mol,
    computer_type = 'fock'
    )

alg_fock.run(
    s=s,
    dt=dt
)
print(f'\n\n Efci:   {mol.fci_energy:+12.10f}')


alg_fci = qf.SRQK(
    mol,
    computer_type = 'fci'
    )

alg_fci.run(
    s=s,
    dt=dt
    )

print(f'\n\n Efci:   {mol.fci_energy:+12.10f}')
