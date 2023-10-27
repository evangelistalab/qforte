import qforte as qf
r = 0.7

geom = [
    ('H', (0., 0., 0.0*r)), 
    ('H', (0., 0., 1.0*r)),
    ('H', (0., 0., 2.0*r)),
    ('H', (0., 0., 3.0*r)),
    ('H', (0., 0., 4.0*r)), 
    ('H', (0., 0., 5.0*r)),
    ('H', (0., 0., 6.0*r)),
    ('H', (0., 0., 7.0*r)),
    ('H', (0., 0., 8.0*r)),
    ('H', (0., 0., 9.0*r))
    ]

timer = qf.local_timer()

timer.reset()
mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g',
    run_fci=1)
timer.record("mol build")


# timer.reset()
# alg_fock = qf.UCCNPQE(
#     mol,
#     computer_type = 'fock'
#     )
# timer.record("alg setup fock")

# timer.reset()
# alg_fock.run(opt_thresh=1.0e-2, pool_type='SD')
# timer.record("run alg fock")

timer.reset()
alg_fci = qf.UCCNPQE(
    mol,
    computer_type = 'fci',
    verbose=False)
timer.record("alg setup fci")


timer.reset()
alg_fci.run(opt_thresh=1.0e-2, pool_type='SDTQ')
timer.record("run alg fci")


print(f' Efci:    {mol.fci_energy:+12.10f}')

print("\n Total Script Time \n")
print(timer)


