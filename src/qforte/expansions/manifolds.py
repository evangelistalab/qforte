"""
Quality-of-life shortcuts to build manifolds for excited state/addition/electron ionization methods
"""

import copy
from itertools import combinations
import qforte as qf
from qforte import build_refprep
from pytest import approx
import numpy as np


def ee_ip_ea_manifold(
    ref, n_ann, n_cre, sz=[0], irreps=None, target_irrep=None, ignore_spin_pairs=False
):
    """Compute all excitations where n_ann occupied orbitals are annihilated
    and n_cre virtual electrons are created
    sz is a list of allowed spin changes.
    (e.g. sz = -1 would allow i->ab')
    irreps is a list of the irreducible representations of each spatial orbital.
    (Defaults to all 0's, i.e. C1)

    target_irrep is the target irreducible representation.  If None, no symmetry restriction.
    ignore_spin_pairs indicates that only determinants unique up to spin-complementation will be included.
    """
    if irreps == None:
        print("No irreps specified for manifold.  Assuming C1 symmetry.")
        irreps = [0] * len(ref)
    occs = [i for i in range(len(ref)) if ref[i] == 1]
    noccs = [i for i in range(len(ref)) if ref[i] == 0]

    annihilate = combinations(occs, n_ann)
    create = combinations(noccs, n_cre)

    dets = []

    for ann in annihilate:
        for cre in copy.copy(create):
            net_spin = 0
            for i in ann:
                if i % 2 == 0:
                    net_spin -= 0.5
                else:
                    net_spin += 0.5

            for a in cre:
                if a % 2 == 0:
                    net_spin += 0.5
                else:
                    net_spin -= 0.5

            if approx(net_spin, abs=1e-10) in sz:
                new_det = copy.deepcopy(ref)
                for i in ann:
                    new_det[i] = 0
                for a in cre:
                    new_det[a] = 1

                if (target_irrep == None) or qf.find_irrep(
                    irreps, [k for k in range(len(new_det)) if new_det[k] == 1]
                ) == target_irrep:
                    if ignore_spin_pairs:
                        sc = [0] * len(new_det)
                        for j in range(int(len(new_det) / 2)):
                            sc[2 * j] = new_det[2 * j + 1]
                            sc[2 * j + 1] = new_det[2 * j]
                        if sc in dets:
                            continue
                    dets.append(new_det)
    return dets


def cin_manifold(
    ref, N, sz=[0], irreps=None, target_irrep=None, ignore_spin_pairs=False
):
    """
    Compute all the N-electron excitations away from an occupation list ref.  (Does not assume HF.)
    """
    return ee_ip_ea_manifold(
        ref,
        N,
        N,
        sz=sz,
        irreps=irreps,
        target_irrep=target_irrep,
        ignore_spin_pairs=ignore_spin_pairs,
    )


def cis_manifold(ref, sz=[0], irreps=None, target_irrep=None, spin_adapt=False):
    """
    Get the singly excited dets.
    """
    return cin_manifold(
        ref,
        1,
        sz=sz,
        irreps=irreps,
        target_irrep=target_irrep,
        ignore_spin_pairs=spin_adapt,
    )


def cisd_manifold(ref, sz=[0], irreps=None, target_irrep=None):
    """
    Get the singly and doubly excited dets.
    """
    return cin_manifold(
        ref, 1, sz=sz, irreps=irreps, target_irrep=target_irrep
    ) + cin_manifold(ref, 2, sz=sz, irreps=irreps, target_irrep=target_irrep)


def sc_double(ref, p, q, r, s, sign):
    """
    Get a specific, spin-complemented double excitation (a_pq^rs (+/-) a_(p'q')^(r's'))|ref>.  Mult determines the sign.
    """
    print("WARNING: Limited Testing of SC Doubles")
    pa, pb, qa, qb, ra, rb, sa, sb = (
        2 * p,
        2 * p + 1,
        2 * q,
        2 * q + 1,
        2 * r,
        2 * r + 1,
        2 * s,
        2 * s + 1,
    )
    diff = [0] * len(ref)
    diff[pa] = -1
    diff[qa] = -1
    diff[ra] = 1
    diff[sa] = 1

    U = qf.Circuit()

    for i in range(len(ref)):
        if ref[i] == 1 and i not in [p, q, r, s]:
            U.add(qf.gate("X", i, i))
    if sign == "minus":
        U.add(qf.gate("X", pa))
    else:
        try:
            assert sign == "plus"
        except:
            print("Options are + and -")
    U.add(qf.gate("H", pa))
    U.add(qf.gate("CNOT", pb, pa))
    U.add(qf.gate("CNOT", qa, pa))
    U.add(qf.gate("CNOT", qb, pb))
    U.add(qf.gate("CNOT", ra, pa))
    U.add(qf.gate("CNOT", rb, pb))
    U.add(qf.gate("CNOT", sa, qa))
    U.add(qf.gate("CNOT", sb, qb))
    U.add(qf.gate("X", pb))
    U.add(qf.gate("X", qb))
    U.add(qf.gate("X", ra))
    U.add(qf.gate("X", sa))

    return U


def sa_single(ref, p, q, mult=1):
    """
    Get a specific, spin-adapted single excitation a_p^q|ref>.  Mult determines the singlet or triplet CSF.
    """

    pa, pb, qa, qb = 2 * p, 2 * p + 1, 2 * q, 2 * q + 1
    diff = [0] * len(ref)
    diff[pa] = -1
    diff[qa] = 1
    new_det = [ref[i] + diff[i] for i in range(len(ref))]

    U = qf.Circuit()
    do_j = []
    if mult == 1:
        for k in range(int(len(ref) / 2)):
            if new_det[2 * k] == 1 or new_det[2 * k + 1] == 1 or 2 * k == pa:
                do_j.append(2 * k)
                if 2 * k + 1 not in [pb, qb]:
                    do_j.append(2 * k + 1)
        for j in do_j:
            U.add(qf.gate("X", j, j))
        U.add(qf.gate("H", pa, pa))
        U.add(qf.gate("CNOT", pb, pa))
        U.add(qf.gate("X", pa, pa))
        U.add(qf.gate("CNOT", qa, pa))
        U.add(qf.gate("CNOT", qb, pa))

    elif mult == 3:
        for k in range(int(len(ref) / 2)):
            if new_det[2 * k] == 1 or new_det[2 * k + 1] == 1:
                if 2 * k != pa:
                    do_j.append(2 * k)
                if 2 * k + 1 not in [pb, qb]:
                    do_j.append(2 * k + 1)
        for j in do_j:
            U.add(qf.gate("X", j, j))
        U.add(qf.gate("H", pa, pa))
        U.add(qf.gate("CNOT", pb, pa))
        U.add(qf.gate("X", pa, pa))
        U.add(qf.gate("CNOT", qa, pa))
        U.add(qf.gate("CNOT", qb, pa))
    else:
        print("Invalid multiplicity.")
        exit()
    return U


def sa_cis(ref, sz=[0], mult=[1, 3], irreps=None, target_irrep=None):
    """
    Get list of unitaries to get singlet and/or triplet states (|a>+|b>, |a>-|b>)
    """
    s = np.sum([ref[j] * (j % 2 - 0.5) for j in range(len(ref))])
    try:
        assert abs(s) < 1e-10 and sz == [0]
    except:
        print("Spin flips not supported.")
        exit()
    dets = cis_manifold(
        ref, sz=sz, irreps=irreps, target_irrep=target_irrep, spin_adapt=True
    )
    Us = []
    for det in dets:
        sc = [0] * len(dets)
        for j in range(int(len(dets) / 2)):
            sc[2 * j + 1] = det[2 * j]
            sc[2 * j] = det[2 * j + 1]
        if sc == det:
            Us.append(build_refprep(det, "occupation_list"))
        else:
            diff = [det[i] - ref[i] for i in range(len(det))]
            inds = [int(diff.index(-1) / 2), int(diff.index(1) / 2)]
            pa, pb, qa, qb = 2 * inds[0], 2 * inds[0] + 1, 2 * inds[1], 2 * inds[1] + 1
            if 1 in mult:
                do_j = []
                U = qf.Circuit()
                for k in range(int(len(ref) / 2)):
                    if det[2 * k] == 1 or det[2 * k + 1] == 1 or 2 * k == pa:
                        do_j.append(2 * k)
                        if 2 * k + 1 not in [pb, qb]:
                            do_j.append(2 * k + 1)
                for j in do_j:
                    U.add(qf.gate("X", j, j))
                U.add(qf.gate("H", pa, pa))
                U.add(qf.gate("CNOT", pb, pa))
                U.add(qf.gate("X", pa, pa))
                U.add(qf.gate("CNOT", qa, pa))
                U.add(qf.gate("CNOT", qb, pa))
                Us.append(U)

            if 3 in mult:
                do_j = []
                U = qf.Circuit()
                for k in range(int(len(ref) / 2)):
                    if det[2 * k] == 1 or det[2 * k + 1] == 1:
                        if 2 * k != pa:
                            do_j.append(2 * k)
                        if 2 * k + 1 not in [pb, qb]:
                            do_j.append(2 * k + 1)
                for j in do_j:
                    U.add(qf.gate("X", j, j))
                U.add(qf.gate("H", pa, pa))
                U.add(qf.gate("CNOT", pb, pa))
                U.add(qf.gate("X", pa, pa))
                U.add(qf.gate("CNOT", qa, pa))
                U.add(qf.gate("CNOT", qb, pa))
                Us.append(U)

    return Us
