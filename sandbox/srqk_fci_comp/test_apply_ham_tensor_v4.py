# This tests the new 2nd order trotter function, 
# there is no actyal 'fci_new' computer type, its a temporary hack 
# used to test this function agains the old one


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
    # ('H', (0., 0., 9.00)),
    # ('H', (0., 0., 10.00))
    ]

mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g',
    run_fci=1)


s = 8
dt = 0.01

r = 1
order = 2

apply_ham_as_tensor = True


alg_fci = qf.SRQK(
    mol,
    computer_type = 'fci',
    trotter_number=r,
    trotter_order=order,
    use_exact_evolution=False,
    apply_ham_as_tensor=apply_ham_as_tensor,
    )

timer = qf.local_timer()
timer.reset()

alg_fci.run(
    s=s,
    dt=dt
    )

timer.record(f"Run SRQK FCI")

print(f"\n\nApply ham as tensor: {apply_ham_as_tensor}")
print(timer)

Eold = alg_fci.get_gs_energy()

print('\n\n')
print(f' Efci:    {mol.fci_energy:+12.10f}')
print(f' Eold:    {Eold:+12.10f}')
print(f' Eold:    {Eold-mol.fci_energy:+12.10f}')


#LGTM!

