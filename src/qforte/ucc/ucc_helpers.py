import qforte

# NOTE: elimenate general ccsd or ucc stuff

def get_ucc_zeros_lists(nocc, nvir, order=2, make_anti_herm=True):

    norb = nocc + nvir
    sq_excitations = []

    if(order < 1):
        raise ValueError("Coupled Cluster excitation order must be at least 1 (CCS)")

    if(order > 0):
        if(make_anti_herm):
            sq_excitations.extend( list( chain.from_iterable(
                                    ( [(a, i), 0.0], [(i, a), -0.0] )
                                        for i in range(nocc)
                                        for a in range (nocc, norb)
                                        ) ) )

        else:
            sq_excitations.extend( [ [(a, i), 0.0]
                                    for i in range(nocc)
                                    for a in range (nocc, norb)
                                    ] )

        if(order > 1):
            if(make_anti_herm):
                sq_excitations.extend( list( chain.from_iterable(
                                        ( [(a, b, j, i), 0.0], [(i, j, b, a), -0.0] )
                                            for i in range(nocc)
                                            for j in range (i+1, nocc)
                                            for a in range(nocc, norb)
                                            for b in range (a+1, norb)
                                            ) ) )


            else:
                sq_excitations.extend( [ [(a, b, j, i), 0.0 ]
                                        for i in range(nocc)
                                        for j in range (i+1, nocc)
                                        for a in range(nocc, norb)
                                        for b in range (a+1, norb)
                                        ] )

            if(order > 2):
                if(make_anti_herm):
                    sq_excitations.extend( list( chain.from_iterable(
                                            ( [(a, b, c, k, j, i), 0.0], [(i, j, k, c, b, a), -0.0] )
                                                for i in range(nocc)
                                                for j in range (i+1, nocc)
                                                for k in range (j+j, nocc)
                                                for a in range(nocc, norb)
                                                for b in range (a+1, norb)
                                                for c in range (b+1, norb)
                                                ) ) )
                else:
                    sq_excitations.extend( [ [(a, b, c, k, j, i), 0.0 ]
                                            for i in range(nocc)
                                            for j in range (i+1, nocc)
                                            for k in range (j+j, nocc)
                                            for a in range(nocc, norb)
                                            for b in range (a+1, norb)
                                            for c in range (b+1, norb)
                                            ] )
                if(order > 3):
                    raise ValueError("QForte currently only supports up to CCSDT")

    return sq_excitations

def get_uccsd_from_ccsd(nocc, nvir, ccsd_singles, ccsd_doubles, include_zero_amps=False, make_anti_herm=True):
    #NOTE: this function may not be nesecary?
    # do like open fermion, but also include possible aplitudes that are zero...

    norb = nocc + nvir
    sq_excitations = []
    threshold = 1.0e-9

    if(include_zero_amps):
        if(make_anti_herm):
            sq_excitations.extend( list( chain.from_iterable(
                                    ( [(a, i), ccsd_singles[a,i]], [(i, a), -ccsd_singles[a,i]] )
                                        for i in range(nocc)
                                        for a in range (nocc, norb)
                                        ) ) )

        else:
            sq_excitations.extend( [ [(a, i), ccsd_singles[a,i]]
                                    for i in range(nocc)
                                    for a in range (nocc, norb)
                                    ] )

                                #### HERE!!! ######

        if(make_anti_herm):
            sq_excitations.extend( list( chain.from_iterable(
                                    ( [(a, b, j, i), ccsd_doubles[a, i, b, j]], [(i, j, b, a), -ccsd_doubles[a, i, b, j]] )
                                        for i in range(nocc)
                                        for j in range (i+1, nocc)
                                        for a in range(nocc, norb)
                                        for b in range (a+1, norb)
                                        ) ) )

        else:
            sq_excitations.extend( [ [(a, b, j, i), ccsd_doubles[a, i, b, j] ]
                                    for i in range(nocc)
                                    for j in range (i+1, nocc)
                                    for a in range(nocc, norb)
                                    for b in range (a+1, norb)
                                    ] )

    else:
        for i, j in zip(*ccsd_singles.nonzero()):
            sq_excitations.append([(i, j), ccsd_singles[i, j] ])
            if(make_anti_herm):
                sq_excitations.append([(j, i), -ccsd_singles[i, j] ])

        for i, j, k, l in zip(*ccsd_doubles.nonzero()):
            sq_excitations.append([(i, k, l, j), ccsd_doubles[i, j, k, l]])
            if(make_anti_herm):
                sq_excitations.append([(j, l, k, i), -ccsd_doubles[i, j, k, l]])


    return sq_excitations

# ##############################################################################
# ##############################################################################
# def get_ucc_zeros_lists(nocc, nvir, order=2):
#
#     norb = nocc + nvir
#     sq_excitations = []
#
#     if(order < 1):
#         raise ValueError("Coupled Cluster excitation order must be at least 1 (CCS)")
#
#     if(order > 0):
#         sq_excitations.extend( [ [(a, i), 0.0]
#                                 for i in range(nocc)
#                                 for a in range (nocc, norb) ])
#
#         if(order > 1):
#             sq_excitations.extend( [ [(a, b, j, i), 0.0 ]
#                                     for i in range(nocc)
#                                     for j in range (i+1, nocc)
#                                     for a in range(nocc, norb)
#                                     for b in range (a+1, norb) ] )
#
#             if(order > 2):
#                 sq_excitations.extend( [ [(a, b, j, i), 0.0 ]
#                                         for i in range(nocc)
#                                         for j in range (i+1, nocc)
#                                         for k in range (j+j, nocc)
#                                         for a in range(nocc, norb)
#                                         for b in range (a+1, norb)
#                                         for c in range (b+1, norb) ] )
#                 if(order > 3):
#                     raise ValueError("QForte currently only supports up to CCSDT")
#
#     return sq_excitations
#
# def get_uccsd_from_ccsd(nocc, nvir, ccsd_singles, ccsd_doubles, include_zero_amps=False, make_anti_herm=True):
#     # make list like open fermion, but also include possible aplitudes that are zero...
#
#     norb = nocc + nvir
#     sq_excitations = []
#     threshold = 1.0e-9
#
#     if(include_zero_amps):
#
#         sq_excitations.extend( [ [(a, i), ccsd_singles[a,i]]
#                                 for i in range(nocc)
#                                 for a in range (nocc, norb) ])
#
#         # NOT ANIT HERMITIAN YET!
#         # if(make_anti_herm):
#         #     sq_excitations.append( [ [(i, j), ccsd_singles[i,j]] for i in range(norb)
#         #                                                          for j in range (norb)])
#
#         sq_excitations.extend( [ [(a, b, j, i), ccsd_doubles[a, i, b, j] ]
#                                 for i in range(nocc)
#                                 for j in range (i+1, nocc)
#                                 for a in range(nocc, norb)
#                                 for b in range (a+1, norb) ] )
#
#         # NOT ANIT HERMITIAN YET!
#         # if(make_anti_herm):
#         #     sq_excitations.append( [ [(i, j, k, l), ccsd_doubles[i,k,l,j]]
#         #                                             for i in range(norb) for j in range (norb)
#         #                                             for k in range(norb) for l in range (norb)])
#
#         ## sq_excitations.extend( [ [(b, a, j, i), ccsd_doubles[b, i, a, j] ]
#         ##                         for i in range(nocc)
#         ##                         for j in range (i+1, nocc)
#         ##                         for a in range(nocc, norb)
#         ##                         for b in range (a+1, norb) ] )
#         ##
#         ## sq_excitations.extend( [ [(b, a, i, j), ccsd_doubles[b, j, a, i] ]
#         ##                         for i in range(nocc)
#         ##                         for j in range (i+1, nocc)
#         ##                         for a in range(nocc, norb)
#         ##                         for b in range (a+1, norb) ] )
#         ##
#         ## sq_excitations.extend( [ [(a, b, i, j), ccsd_doubles[a, j, b, i] ]
#         ##                         for i in range(nocc)
#         ##                         for j in range (i+1, nocc)
#         ##                         for a in range(nocc, norb)
#         ##                         for b in range (a+1, norb) ] )
#
#
#
#     else:
#         for i, j in zip(*ccsd_singles.nonzero()):
#             sq_excitations.append([(i, j), ccsd_singles[i, j] ])
#             if(make_anti_herm):
#                     sq_excitations.append([(j, i), -ccsd_singles[i, j] ])
#
#         for i, j, k, l in zip(*ccsd_doubles.nonzero()):
#             sq_excitations.append([(i, k, l, j), ccsd_doubles[i, j, k, l]])
#             if(make_anti_herm):
#                 sq_excitations.append([(j, l, k, i), -ccsd_doubles[i, j, k, l]])
#
#
#     return sq_excitations
# ##############################################################################
# ##############################################################################

def get_singlet_ucc_zeros_lists(nocc, nvir, order=2, make_anti_herm=True):

    norb = nocc + nvir
    sq_excitations = []

    if(order < 1):
        raise ValueError("Coupled Cluster excitation order must be at least 1 (CCS)")

    if(order > 0):
        if(make_anti_herm):
            sq_excitations.extend( list( chain.from_iterable(
                                    ( [(a, i), 0.0], [(i, a), -0.0] )
                                        for i in range(nocc)
                                        for a in range (nocc, norb)
                                        if cse((a,i)) ) ) )

        else:
            sq_excitations.extend( [ [(a, i), 0.0]
                                    for i in range(nocc)
                                    for a in range (nocc, norb)
                                    if cse((a,i)) ])

        if(order > 1):
            if(make_anti_herm):
                sq_excitations.extend( list( chain.from_iterable(
                                        ( [(a, b, j, i), 0.0], [(i, j, b, a), -0.0] )
                                            for i in range(nocc)
                                            for j in range (nocc) if (j!=i)
                                            for a in range(nocc, norb)
                                            for b in range (nocc, norb) if (b!=a)
                                            if cse((a, b, j, i)) ) ) )


            else:
                sq_excitations.extend( [ [(a, b, j, i), 0.0 ]
                                        for i in range(nocc)
                                        for j in range (nocc) if (j!=i)
                                        for a in range(nocc, norb)
                                        for b in range (nocc, norb) if (b!=a)
                                        if cse((a, b, j, i)) ] )

            if(order > 2):
                if(make_anti_herm):
                    sq_excitations.extend( list( chain.from_iterable(
                                            ( [(a, b, c, k, j, i), 0.0], [(i, j, k, c, b, a), -0.0] )
                                                for i in range(nocc)
                                                for j in range (nocc) if (j!=i)
                                                for k in range (nocc) if ((k!=j) and (k!=i))
                                                for a in range(nocc, norb)
                                                for b in range (nocc, norb) if (b!=a)
                                                for c in range (nocc, norb) if ((c!=b) and (c!=a))
                                                if cse((a, b, c, k, j, i)) ) ) )
                else:
                    sq_excitations.extend( [ [(a, b, c, k, j, i), 0.0 ]
                                            for i in range(nocc)
                                            for j in range (nocc) if (j!=i)
                                            for k in range (nocc) if ((k!=j) and (k!=i))
                                            for a in range(nocc, norb)
                                            for b in range (nocc, norb) if (b!=a)
                                            for c in range (nocc, norb) if ((c!=b) and (c!=a))
                                            if cse((a, b, c, k, j, i)) ] )
                if(order > 3):
                    raise ValueError("QForte currently only supports up to CCSDT")

    return sq_excitations

def get_singlet_uccsd_from_ccsd(nocc, nvir, ccsd_singles, ccsd_doubles, include_zero_amps=False, make_anti_herm=True):
    #NOTE: this function may not be nesecary?
    # do like open fermion, but also include possible aplitudes that are zero...

    norb = nocc + nvir
    sq_excitations = []
    threshold = 1.0e-9

    if(include_zero_amps):
        if(make_anti_herm):
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

        if(make_anti_herm):
            sq_excitations.extend( list( chain.from_iterable(
                                    ( [(a, b, j, i), ccsd_doubles[a, i, b, j]], [(i, j, b, a), -ccsd_doubles[a, i, b, j]] )
                                        for i in range(nocc)
                                        for j in range (i+1, nocc)
                                        for a in range(nocc, norb)
                                        for b in range (a+1, norb)
                                        if cse((a, b, j, i)) ) ) )

        else:
            sq_excitations.extend( [ [(a, b, j, i), ccsd_doubles[a, i, b, j] ]
                                    for i in range(nocc)
                                    for j in range (i+1, nocc)
                                    for a in range(nocc, norb)
                                    for b in range (a+1, norb)
                                    if cse((a, b, j, i)) ] )

    else:
        for i, j in zip(*ccsd_singles.nonzero()):
            if(cse(i,j)):
                sq_excitations.append([(i, j), ccsd_singles[i, j] ])
                if(make_anti_herm):
                    sq_excitations.append([(j, i), -ccsd_singles[i, j] ])

        for i, j, k, l in zip(*ccsd_doubles.nonzero()):
            if(cse(i, k, l, j)):
                sq_excitations.append([(i, k, l, j), ccsd_doubles[i, j, k, l]])
                if(make_anti_herm):
                    sq_excitations.append([(j, l, k, i), -ccsd_doubles[i, j, k, l]])


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
