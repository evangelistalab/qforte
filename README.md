Qforte
==============================
[//]: # (Badges)

![Travis Build Status](https://travis-ci.org/evangelistalab/qforte.svg?branch=master)


Qforte is an open-source quantum computer simulator and algorithms library for molecular simulation.

Install Dependancies (Recommended)
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

# Construct a Bell state
computer = qforte.QuantumComputer(2)
computer.apply_gate(qforte.make_gate('H',0,0))
computer.apply_gate(qforte.make_gate('cX',1,0))

```

### Copyright

Copyright (c) 2019, The Evangelista Lab
