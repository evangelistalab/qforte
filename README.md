Qforte
==============================
[//]: # (Badges)

[![Travis Build Status](https://travis-ci.org/evangelistalab/qforte.svg?branch=master)


Qforte is an open-source quantum computer simulator and algorithms library for molecular simulation.

Installation
------------

```bash
git clone https://github.com/evangelistalab/qforte.git
cd qforte
python setup.py build develop
```

#### run tests:
```bash
cd tests/
pytest -v
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

