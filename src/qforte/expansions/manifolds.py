"""
Quality-of-life shortcuts to build manifolds for excited state/addition/electron ionization methods
"""
import copy
from itertools import combinations
import qforte as qf
from pytest import approx

def ee_ip_ea_manifold(ref, n_ann, n_cre, sz, irreps):
    """Compute all excitations where n_ann occupied orbitals are annihilated
      and n_cre virtual electrons are created
      sz is a list of allowed spin changes.
      (e.g. sz = -1 would allow i->ab')
      irreps is a list of the irreducible representations of each spatial orbital.
      """
    if(irreps==None):
        print("No irreps specified for manifold.  Assuming C1 symmetry.")
        irreps = [0]*len(ref)
    
    occs = [i for i in range(len(ref)) if ref[i] == 1]
    noccs = [i for i in range(len(ref)) if ref[i] == 0]
    
    annihilate = combinations(occs, n_ann)
    create = combinations(noccs, n_cre)
    
    dets = []

    for ann in annihilate:
        for cre in copy.copy(create):
            net_spin = 0
            for i in ann:
                if i%2 == 0:
                    net_spin -= .5
                else:
                    net_spin += .5
                
            for a in cre:
                if a%2 == 0:
                    net_spin += .5
                else:
                    net_spin -= .5
                
            if approx(net_spin, abs = 1e-10) in sz and (qf.find_irrep(irreps, [k for k in (ann + cre)]) == 0):
                new_det = copy.deepcopy(ref)
                for i in ann:
                    new_det[i] = 0
                for a in cre:
                    new_det[a] = 1
                dets.append(new_det)
    return dets

def cin_manifold(ref, N, irreps = None):
    """
    Compute all the N-electron excitations away from an occupation list ref.  (Does not assume HF.)
    """
    return ee_ip_ea_manifold(ref, N, N, [0], irreps)

def cis_manifold(ref, irreps = None):
    """
    Get the singly excited dets.
    """
    return cin_manifold(ref, 1, irreps = irreps)

def cisd_manifold(ref, irreps = None):
    """
    Get the singly and doubly excited dets.
    """
    return cin_manifold(ref, 1, irreps = irreps) + cin_manifold(ref, 2, irreps = irreps)

