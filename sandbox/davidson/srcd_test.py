import qforte
from qforte import *
# Define the reference and geometry lists.
geom = [('Li', (0., 0., 0.0)), ('H', (0., 0., 1.50))]

# Get the molecule object that now contains both the fermionic and qubit Hamiltonians.
LiHmol = system_factory(build_type='psi4', mol_geometry=geom, basis='sto-3g')

# alg = SRQK(LiHmol, computer_type='fci')
alg = SRCD(LiHmol, computer_type='fci')
alg.run()