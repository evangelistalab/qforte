"""
Quality-of-life shortcuts to build manifolds for excited state methods
"""
import copy
from itertools import combinations

def ipea_manifold(ref, n_ann, n_cre, sz_delta, irreps = None):
    """Compute all excitations where n occupied orbitals are annihilated
      and n_cre virtual electrons are created
      sz_delta is a list of allowed spin changes.
      (e.g. sz_delta = -1 would allow i->ab')
      irreps is a list of the irreducible representation of each spatial orbital.
      """
    
    M = len(ref)
    occs = [i for i in range(M) if ref[i] == 1]
    noccs = [i for i in range(M) if ref[i] == 0]
    annihilate = combinations(occs, n_ann)
    create = combinations(noccs, n_cre)
    dets = []

    for ann in annihilate:
        for cre in copy.copy(create):
            net_spin = 0
            for i in ann:
                if i%2 == 0:
                    net_spin -= 1
                else:
                    net_spin += 1
            for a in cre:
                if a%2 == 0:
                    net_spin += 1
                else:
                    net_spin -= 1
            if net_spin in sz_delta:
                new_det = copy.deepcopy(ref)
                for i in ann:
                    new_det[i] = 0
                for a in cre:
                    new_det[a] = 1
                dets.append(new_det)

        if (irreps!=None):
            print("Hello")
        exit()
    return dets

def cin_manifold(ref, N):
    """
    Compute all the N-electron excitations away from an occupation list ref.  (Does not assume HF.)
    """
    return ipea_manifold(ref, N, N, [0])

def cis_manifold(ref):
    """
    Get the singly excited dets.
    """
    return cin_manifold(ref, 1)

def cisd_manifold(ref):
    """
    Get the singly and doubly excited dets.
    """
    return cin_manifold(ref, 1) + cin_manifold(ref, 2)

def ip_sd_manifold(ref):
    """
    Get the IP operators used in the Asthana q-sc-EOM paper
    """
    return ipea_manifold(ref, 1, 0, [-1, 1]) + ipea_manifold(ref, 2, 1, [-1, 1])

def ea_sd_manifold(ref):
    """
    Get the EA operators used in the Asthana q-sc-EOM paper
    """
    return ipea_manifold(ref, 0, 1, [-1, 1]) + ipea_manifold(ref, 1, 2, [-1, 1])
