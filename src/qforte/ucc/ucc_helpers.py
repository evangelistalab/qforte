import qforte

def get_ucc_zeros_lists(norb, order=2):

    # the maximum cc excitation order
    # order = 2
    # norb = 4

    sq_excitations = []

    #### Build with all zeros

    if(order < 1):
        raise ValueError("Coupled Cluster excitation order must be at least 1 (CCS)")

    if(order > 0):
        sq_excitations.append([[(i, j), 0.0] for i in range(norb) for j in range (norb)])

        if(order > 1):
            sq_excitations.append([[(i, j, k, l), 0.0] for i in range(norb) for j in range (norb)
                                                       for k in range(norb) for l in range (norb)])

            if(order > 2):
                sq_excitations.append([[(i, j, k, l, m, n), 0.0] for i in range(norb) for j in range (norb)
                                                     for k in range(norb) for l in range (norb)
                                                     for m in range(norb) for n in range (norb)])
                if(order > 3):
                    raise ValueError("QForte currently only supports up to CCSDT")

    return sq_excitations

def get_uccsd_from_ccsd(norb, ccsd_singles, ccsd_doubles, include_zero_amps=False, make_anit_herm=True):
    # do like open fermion, but also include possible aplitudes that are zero...
    sq_excitations = []

    threshold = 1.0e-9

    if(include_zero_amps):

        # NOTE: Not yet working
        sq_excitations.append( [ [(i, j), ccsd_singles[i,j]] for i in range(norb)
                                                             for j in range (norb)])

        # NOT ANIT HERMITIAN YET!
        # if(make_anit_herm):
        #     sq_excitations.append( [ [(i, j), ccsd_singles[i,j]] for i in range(norb)
        #                                                          for j in range (norb)])

        # NOTE: Not yet working
        sq_excitations.append( [ [(i, j, k, l), ccsd_doubles[i,k,l,j]]
                                                for i in range(norb) for j in range (norb)
                                                for k in range(norb) for l in range (norb)])

        # NOT ANIT HERMITIAN YET!
        # if(make_anit_herm):
        #     sq_excitations.append( [ [(i, j, k, l), ccsd_doubles[i,k,l,j]]
        #                                             for i in range(norb) for j in range (norb)
        #                                             for k in range(norb) for l in range (norb)])

    else:
        for i, j in zip(*ccsd_singles.nonzero()):
            sq_excitations.append([(i, j), ccsd_singles[i, j] ])
            if(make_anit_herm):
                    sq_excitations.append([(j, i), -ccsd_singles[i, j] ])

        for i, j, k, l in zip(*ccsd_doubles.nonzero()):
            sq_excitations.append([(i, k, l, j), ccsd_doubles[i, j, k, l]])
            if(make_anit_herm):
                sq_excitations.append([(j, l, k, i), -ccsd_doubles[i, j, k, l]])


    return sq_excitations
