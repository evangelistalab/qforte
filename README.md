Qforte
==============================
[//]: # (Badges)

![Travis Build Status](https://travis-ci.org/evangelistalab/qforte.svg?branch=master)


Qforte is an open-source quantum computer simulator and algorithms library for molecular simulation. It includes implementations of the following algorithms: quantum phase estimation (QPE), multireference selected quantum Krylov (MRSQK), quantum imaginary time evolution (QITE), ADAPT variational quantum eigensolver (VQE), and unitary coupled cluster singles and doubles VQE (UCCSD-VQE).

Install Dependencies (Recommended)
----------------------------------

#### create and activate qforte environment:
```bash
conda create -n qforte_env python=3.7
conda activate qforte_env
```

#### install psi4 and openfermion:
```bash
conda install psi4 openfermion openfermionpsi4 -c psi4
```

Installation (For Development)
------------------------------

```bash
git clone https://github.com/evangelistalab/qforte.git
cd qforte
python setup.py develop
```

#### run tests:
```bash
python setup.py test
```

Getting Started
---------------
```python
import qforte

# Construct a Bell state.
computer = qforte.QuantumComputer(2)
computer.apply_gate(qforte.make_gate('H',0,0))
computer.apply_gate(qforte.make_gate('cX',1,0))

# Run quantum phase estimation on H2.
from qforte.qpea.qpe import QPE
from qforte.system import system_factory

H2geom = [('H', (0., 0., 0.)), ('H', (0., 0., 1.50))]
H2ref = [1,1,0,0]

adapter = system_factory(mol_geometry=H2geom)
adapter.run()
H2mol = adapter.get_molecule()

alg = QPE(H2mol, H2ref, trotter_number=2)
alg.run(t = 0.4,
        nruns = 100,
        success_prob = 0.5,
        num_precise_bits = 8)

Egs = alg.get_gs_energy()
```

### Copyright

Copyright (c) 2019, The Evangelista Lab
