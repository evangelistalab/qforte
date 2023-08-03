"""
Quality-of-life shortcuts to build manifolds for excited state/addition/electron ionization methods
"""
import copy
from itertools import combinations
import qforte as qf
from pytest import approx

def ee_ip_ea_manifold(ref, n_ann, n_cre, sz = [0], irreps = None, target_irrep = None):
    """Compute all excitations where n_ann occupied orbitals are annihilated
      and n_cre virtual electrons are created
      sz is a list of allowed spin changes.
      (e.g. sz = -1 would allow i->ab')
      irreps is a list of the irreducible representations of each spatial orbital.
      (Defaults to all 0's, i.e. C1)

      target_irrep is the target irreducible representation.  If None, no symmetry restriction.
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
                
            if approx(net_spin, abs = 1e-10) in sz:
                new_det = copy.deepcopy(ref)
                for i in ann:
                    new_det[i] = 0
                for a in cre:
                    new_det[a] = 1
                
                if (target_irrep==None) or qf.find_irrep(irreps, [k for k in range(len(new_det)) if new_det[k] == 1]) == target_irrep:
                    dets.append(new_det)
    return dets

def cin_manifold(ref, N, sz = [0], irreps = None, target_irrep = None):    
    """
    Compute all the N-electron excitations away from an occupation list ref.  (Does not assume HF.)
    """
    return ee_ip_ea_manifold(ref, N, N, sz = sz, irreps = irreps, target_irrep = target_irrep)

def cis_manifold(ref, sz = [0], irreps = None, target_irrep = None):
    """
    Get the singly excited dets.
    """
    return cin_manifold(ref, 1, sz = sz, irreps = irreps, target_irrep = target_irrep)

def cisd_manifold(ref, sz = [0], irreps = None, target_irrep = None):
    """
    Get the singly and doubly excited dets.
    """
    return cin_manifold(ref, 1, sz = sz, irreps = irreps, target_irrep = target_irrep) + cin_manifold(ref, 2, sz = sz, irreps = irreps, target_irrep = target_irrep)

