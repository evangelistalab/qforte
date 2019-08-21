import qforte
from qforte.vqe.adapt_vqe import *

import pytest
import sys
import numpy as np
import numpy.testing as npt
from timeit import default_timer as timer

def test_build_operator_pool():
    geom = [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))]
    # set_up = {basis, 
    # multiplicity, 
    # charge, 
    # description, 
    # run_scf, 
    # run_mp2, 
    # run_cisd, 
    # run_ccsd, 
    # run_fci }
    mol = runPsi4(geom)
    adapt_obj = ADAPT_VQE(mol)
    adapt_obj.build_operator_pool()
    adapt_ins.jw_ops




