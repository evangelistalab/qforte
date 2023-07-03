"""
Quality-of-life shortcuts to build manifolds for excited state methods
"""
import copy
from itertools import combinations

def cin_manifold(ref, N):
    """
    Compute all the N-electron excitations away from an occupation list ref.  (Does not assume HF.)
    """
    M = len(ref)
    occs = [i for i in range(M) if ref[i] == 1]
    noccs = [i for i in range(M) if ref[i] == 0]
    annihilate = combinations(occs, N)
    create = combinations(noccs, N)
    dets = []
    
    for ann in annihilate:
        print(f"Outer {ann}")
        for cre in copy.copy(create):
            spins_ann = [i%2 for i in ann]
            spins_cre = [a%2 for a in cre] 
            if spins_ann.count(1) != spins_cre.count(1):
                continue
            new_det = copy.deepcopy(ref)
            for i in ann:
                new_det[i] = 0
            for a in cre:
                new_det[a] = 1
            dets.append(new_det)
    return dets

def cisd_manifold(ref):
    """
    Get the singly and doubly excited dets.
    """
    return cin_manifold(ref, 1) + cin_manifold(ref, 2)

