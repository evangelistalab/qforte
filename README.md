Qforte
==============================
[//]: # (Badges)

[![Python Package using Conda](https://github.com/evangelistalab/qforte/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/evangelistalab/qforte/actions/workflows/python-package-conda.yml)
[![codecov](https://codecov.io/gh/evangelistalab/qforte/graph/badge.svg?token=LE8PEQC2BH)](https://codecov.io/gh/evangelistalab/qforte)
[![Documentation Status](https://readthedocs.org/projects/qforte/badge/?version=latest)](https://qforte.readthedocs.io/en/latest/?badge=latest)


QForte is comprehensive development tool for new quantum simulation algorithms that also contains black-box implementations of a wide variety of existing algorithms. 
It incorporates functionality for handling molecular Hamiltonians, fermionic
encoding, automated ansatz construction, time evolution, state-vector simulation, operator averaging, and computational resource estimation.
QForte requires only a classical electronic structure package as a dependency.

Black Box Algorithm Implementations
-----------------------------------
- Disentangled (Trotterized) unitary coupled cluster variational quantum eigensolver (dUCCVQE)
  - QForte will treat up to hex-tuple particle-hole excitations (SDTQPH) or generalized singled and doubles (GSD)


- Adaptive derivative-assembled pseudo Trotterized VQE (ADAPT-VQE)
  
  
- Disentangled (factorized) unitary coupled cluster projective quantum eigensolver (dUCCPQE)
  - QForte will treat up to hex-tuple particle-hole excitations (SDTQPH)
  
  
- Selected projective quantum eigensolver (SPQE)


- Single reference Quantum Krylov diagonalization (SRQK)


- Multireference selected quantum Krylov diagonalization (MRSQK)


- Quantum imaginary time evolution (QITE)


- Quantum Lanczos (QL)


- Pilot implementation of Quantum phase estimation (QPE)


Install Dependencies (Recommended)
----------------------------------

#### create and activate qforte environment:
```bash
conda create -n qforte_env python
conda activate qforte_env
```

#### install required packages:
```bash
conda install psi4 -c psi4
conda install scipy>=1.11
```

Installation (For Development)
------------------------------

```bash
git clone --recurse-submodules https://github.com/evangelistalab/qforte.git
cd qforte
python setup.py develop
```

To supply custom arguments to `cmake` for installation, you can either edit `setup.py` or `CMakeLists.txt`.

#### run tests:
```bash
cd tests
pytest
```

Getting Started
---------------

QForte's state-vector simulator can be used for simple tasks, such as the construction of Bell states, and is the backbone for implementation of all the black-box algorithms. Below are a few examples, more detailed descriptions of QForte's features and algorithms can be found in the release article (https://arxiv.org/abs/2108.04413) and in the Tutorial notebooks.  

```python
import qforte

# Construct a Bell state.
computer = qforte.Computer(2)
computer.apply_gate(qforte.gate('H',0))
computer.apply_gate(qforte.gate('cX',1,0))

## Run black-box algorithms for LiH molecule. ##
from qforte import *

# Define the geometry list.
geom = [('Li', (0., 0., 0.0)), ('H', (0., 0., 1.50))]

# Get the molecule object that now contains the fermionic and qubit Hamiltonians.
LiHmol = system_factory(build_type='psi4', mol_geometry=geom, basis='STO-3g', run_fci=1)

# Run the dUCCSD-VQE algorithm for LiH.
vqe_alg = UCCNVQE(LiHmol)
vqe_alg.run(opt_thresh=1.0e-2, pool_type='SD')

# Run the single reference QK algorithm for LiH.
srqk_alg = SRQK(LiHmol)
srqk_alg.run()

# Get ground state energies predicted by the algorithms, compare to FCI. 
vqe_gs_energy = vqe_alg.get_gs_energy()
srqk_gs_energy = srqk_alg.get_gs_energy()
fci_energy = LiHmol.fci_energy
```

Publications
------------
QForte has been used to implement the novel algorithms presented in the following publications:

> 1. Stair, Nicholas H., and Francesco A. Evangelista. *Simulating Many-Body Systems with a Projective Quantum Eigensolver.* PRX Quantum 2.3 (2021): 030301. https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.030301

> 2. Stair, Nicholas H., Renke Huang, and Francesco A. Evangelista. *A Multireference Quantum Krylov Algorithm for Strongly Correlated Electrons.* Journal of chemical theory and computation 16.4 (2020): 2236-2245. https://pubs.acs.org/doi/10.1021/acs.jctc.9b01125

QForte's release article:

> 3. Stair, Nicholas H., and Francesco A. Evangelista. *Qforte: an efficient state simulator and quantum algorithms library for molecular electronic structure.* arXiv preprint arXiv:2108.04413 (2021). https://arxiv.org/abs/2108.04413

### Copyright

Copyright (c) 2019, The Evangelista Lab
