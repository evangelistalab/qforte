"""
ucc_helpers.py
=================================================
A module to help build operator listst for
unitary coupled cluster.
"""
import qforte

def get_ucc_zeros_lists(nocc, nvir, order=2, make_anti_herm=True):

    norb = nocc + nvir
    sq_excitations = []

    if(order < 1):
        raise ValueError("Coupled Cluster excitation order must be at least 1 (CCS)")

    if(order > 0):
        if(make_anti_herm):
            sq_excitations.extend( list( chain.from_iterable(
                                    ( [(a, i), 0.0], [(i, a), -0.0] )
                                        for i in range (nocc)
                                        for a in range (nocc, norb)
                                        if cse((a,i)) ) ) )

        else:
            sq_excitations.extend( [ [(a, i), 0.0]
                                    for i in range (nocc)
                                    for a in range (nocc, norb)
                                    if cse((a,i)) ])

    if(order > 1):
        if(make_anti_herm):
            sq_excitations.extend( list( chain.from_iterable(
                                    ( [(a, b, j, i), 0.0], [(i, j, b, a), -0.0] )
                                        for i in range (nocc)
                                        for j in range (nocc) if (j!=i)
                                        for a in range (nocc, norb)
                                        for b in range (nocc, norb) if (b!=a)
                                        if cse((a, b, j, i)) ) ) )


        else:
            sq_excitations.extend( [ [(a, b, j, i), 0.0 ]
                                    for i in range (nocc)
                                    for j in range (nocc) if (j!=i)
                                    for a in range (nocc, norb)
                                    for b in range (nocc, norb) if (b!=a)
                                    if cse((a, b, j, i)) ] )

    if(order > 2):
        if(make_anti_herm):
            sq_excitations.extend( list( chain.from_iterable(
                                    ( [(a, b, c, k, j, i), 0.0], [(i, j, k, c, b, a), -0.0] )
                                        for i in range (nocc)
                                        for j in range (nocc) if (j!=i)
                                        for k in range (nocc) if ((k!=j) and (k!=i))
                                        for a in range (nocc, norb)
                                        for b in range (nocc, norb) if (b!=a)
                                        for c in range (nocc, norb) if ((c!=b) and (c!=a))
                                        if cse((a, b, c, k, j, i)) ) ) )
        else:
            sq_excitations.extend( [ [(a, b, c, k, j, i), 0.0 ]
                                    for i in range (nocc)
                                    for j in range (nocc) if (j!=i)
                                    for k in range (nocc) if ((k!=j) and (k!=i))
                                    for a in range (nocc, norb)
                                    for b in range (nocc, norb) if (b!=a)
                                    for c in range (nocc, norb) if ((c!=b) and (c!=a))
                                    if cse((a, b, c, k, j, i)) ] )

    if(order > 3):
        if(make_anti_herm):
            sq_excitations.extend( list( chain.from_iterable(
                                    ( [(a, b, c, d, l, k, j, i), 0.0], [(i, j, k, l, d, c, b, a), -0.0] )
                                        for i in range (nocc)
                                        for j in range (nocc) if (j!=i)
                                        for k in range (nocc) if ((k!=j) and (k!=i))
                                        for l in range (nocc) if ((l!=k) and (l!=j) and (l!=i))
                                        for a in range (nocc, norb)
                                        for b in range (nocc, norb) if (b!=a)
                                        for c in range (nocc, norb) if ((c!=b) and (c!=a))
                                        for d in range (nocc, norb) if ((d!=c) and (d!=b) and (d!=a))
                                        if cse((a, b, c, d, l, k, j, i)) ) ) )
        else:
            sq_excitations.extend( [ [(a, b, c, d, l, k, j, i), 0.0 ]
                                    for i in range (nocc)
                                    for j in range (nocc) if (j!=i)
                                    for k in range (nocc) if ((k!=j) and (k!=i))
                                    for l in range (nocc) if ((l!=k) and (l!=j) and (l!=i))
                                    for a in range (nocc, norb)
                                    for b in range (nocc, norb) if (b!=a)
                                    for c in range (nocc, norb) if ((c!=b) and (c!=a))
                                    for d in range (nocc, norb) if ((d!=c) and (d!=b) and (d!=a))
                                    if cse((a, b, c, d, l, k, j, i)) ] )


    if(order > 4):
        raise ValueError("QForte currently only supports up to CCSDTQ")

    return sq_excitations

def get_ucc_from_ccsd(nocc, nvir, ccsd_singles, ccsd_doubles, order=2, include_zero_amps=False, make_anti_herm=True):

    norb = nocc + nvir
    sq_excitations = []
    threshold = 1.0e-9

    if(order < 1):
        raise ValueError("Coupled Cluster excitation order must be at least 1 (CCS)")

    if(include_zero_amps):
        if( order > 0 ):
            if(make_anti_herm): #Copy starting form bookmarked line above...
                sq_excitations.extend( list( chain.from_iterable(
                                        ( [(a, i), ccsd_singles[a,i]], [(i, a), -ccsd_singles[a,i]] )
                                            for i in range(nocc)
                                            for a in range (nocc, norb)
                                            if cse((a,i)) ) ) )

            else:
                sq_excitations.extend( [ [(a, i), ccsd_singles[a,i]]
                                        for i in range(nocc)
                                        for a in range (nocc, norb)
                                        if cse((a,i))] )

        if( order > 1):
            if(make_anti_herm):
                sq_excitations.extend( list( chain.from_iterable(
                                        ( [(a, b, j, i), ccsd_doubles[a, i, b, j]], [(i, j, b, a), -ccsd_doubles[a, i, b, j]] )
                                            for i in range (nocc)
                                            for j in range (nocc) if (j!=i)
                                            for a in range (nocc, norb)
                                            for b in range (nocc, norb) if (b!=a)
                                            if cse((a, b, j, i)) ) ) )

            else:
                sq_excitations.extend( [ [(a, b, j, i), ccsd_doubles[a, i, b, j] ]
                                        for i in range (nocc)
                                        for j in range (nocc) if (j!=i)
                                        for a in range (nocc, norb)
                                        for b in range (nocc, norb) if (b!=a)
                                        if cse((a, b, j, i)) ] )

        if(order > 2):
            raise ValueError("Using CCSD guess with order > 2 not yet supported.")

    else:
        if( order > 0 ):
            for i, j in zip(*ccsd_singles.nonzero()):
                sq_excitations.append([(i, j), ccsd_singles[i, j] ])
                if(make_anti_herm):
                    sq_excitations.append([(j, i), -ccsd_singles[i, j] ])

        if( order > 1 ):
            for i, j, k, l in zip(*ccsd_doubles.nonzero()):
                sq_excitations.append([(i, k, l, j), ccsd_doubles[i, j, k, l]])
                if(make_anti_herm):
                    sq_excitations.append([(j, l, k, i), -ccsd_doubles[i, j, k, l]])

        if(order > 2):
            raise ValueError("Using CCSD guess with order > 2 not yet supported.")


    return sq_excitations

# check to see if singlet excitation function
def cse(sq_term):
    dn_idx = len(sq_term)-1
    for up_idx in range(int(len(sq_term) / 2)):
        if((sq_term[up_idx] % 2) != (sq_term[dn_idx] % 2)):
            return 0
        else:
            dn_idx -= 1
    return 1
