"""
point_groups.py
====================================================
This module contains information about the irreducible
representations and character tables of the point groups
supported by QForte. Currently, this module pertains
only to molecular objects.
"""

from itertools import islice

def symmetry_irreps(group):
    # This function returns two dictionaries containing the maps
    # irrep --> int and int --> irrep for a given point group

    group = group.lower()

    ## Dictionaries of irreducible representations
    c1_to_int = {
    'A  ':  0
    }

    c1_to_irrep = {
        0 : 'A  '
    }

    c2_to_int = {
        'A  ' : 0,
        'B  ' : 1
    }

    c2_to_irrep = {
        0 : 'A  ',
        1 : 'B  '
    }

    ci_to_int = {
        'Ag ' : 0,
        'Au ' : 1
    }

    ci_to_irrep = {
        0 : 'Ag ',
        1 : 'Au '
    }

    cs_to_int = {
        'Ap ' : 0,
        'App'  : 1
    }

    cs_to_irrep = {
        0 : 'Ap ',
        1 : 'App'
    }

    d2_to_int = {
        'A  ' : 0,
        'B1 ' : 1,
        'B2 ' : 2,
        'B3 ' : 3
    }

    d2_to_irrep = {
        0 : 'A  ',
        1 : 'B1 ',
        2 : 'B2 ',
        3 : 'B3 '
    }

    c2h_to_int = {
        'Ag ' : 0,
        'Bg ' : 1,
        'Au ' : 2,
        'Bu ' : 3
    }

    c2h_to_irrep = {
        0 : 'Ag ',
        1 : 'Bg ',
        2 : 'Au ',
        3 : 'Bu '
    }

    c2v_to_int = {
        'A1 ' : 0,
        'A2 ' : 1,
        'B1 ' : 2,
        'B2 ' : 3
    }

    c2v_to_irrep = {
        0 : 'A1 ',
        1 : 'A2 ',
        2 : 'B1 ',
        3 : 'B2 '
    }

    d2h_to_int = {
        'Ag ' : 0,
        'B1g' : 1,
        'B2g' : 2,
        'B3g' : 3,
        'Au ' : 4,
        'B1u' : 5,
        'B2u' : 6,
        'B3u' : 7
    }

    d2h_to_irrep = {
        0 : 'Ag ',
        1 : 'B1g',
        2 : 'B2g',
        3 : 'B3g',
        4 : 'Au ',
        5 : 'B1u',
        6 : 'B2u',
        7 : 'B3u'
    }

    if group == 'c1':
        irrep_to_int = c1_to_int
        int_to_irrep = c1_to_irrep
    elif group == 'c2':
        irrep_to_int = c2_to_int
        int_to_irrep = c2_to_irrep
    elif group == 'ci':
        irrep_to_int = ci_to_int
        int_to_irrep = ci_to_irrep
    elif group == 'cs':
        irrep_to_int = cs_to_int
        int_to_irrep = cs_to_irrep
    elif group == 'd2':
        irrep_to_int = d2_to_int
        int_to_irrep = d2_to_irrep
    elif group == 'c2h':
        irrep_to_int = c2h_to_int
        int_to_irrep = c2h_to_irrep
    elif group == 'c2v':
        irrep_to_int = c2v_to_int
        int_to_irrep = c2v_to_irrep
    elif group == 'd2h':
        irrep_to_int = d2h_to_int
        int_to_irrep = d2h_to_irrep
    else:
        raise ValueError('The given point group is not supported. Choose one of:\nC1, C2, Ci, Cs, D2, C2h, C2v, D2h')

    return irrep_to_int, int_to_irrep

def char_table(group):

    # Function that prints the character table of a chosen point group

    group = group.lower()

    ## Retrieve the dictionaries of irreps for the given point group
    irrep_to_int, int_to_irrep = symmetry_irreps(group)

    print('==========>', group, '<==========')
    print('')
    print('      ', end = '')

    for irrep in irrep_to_int.items():
        print(irrep[0], ' ', end = '')

    print('\n')

    for irrep1 in irrep_to_int.items():
        print(irrep1[0], '  ', end = '')
        for irrep2 in irrep_to_int.items():
            print(int_to_irrep[irrep1[1] ^ irrep2[1]], ' ', end = '')
        print()

    return None

def symmetry_check(orb_irreps_to_int, annihilate, create):
    sym = 0
    for spinorb in annihilate + create:
        sym ^= orb_irreps_to_int[int(spinorb/2)]

    return sym

def psi4_symmetry(filename):

    # This function parses the output of a Psi4 computation and extracts
    # the symmetry irreducible representations for each spatial molecular
    # orbital.

    rhf = False

    p4_output = open(filename, 'r')

    ## Parse the output and extract information about the system
    ## and the location of the lists of molecular orbitals
    line_nmbr = 0
    for line in p4_output:
        if 'Molecular point group:' in line:
            x = line.split(':')
            pnt_grp = x[1].strip()
        if 'Multiplicity' in line:
            x = line.split('=')
            if int(x[1]) == 1:
                rhf = True
        if 'Doubly Occupied' in line:
            dbl_start = line_nmbr + 1
        if 'Singly Occupied' in line:
            sngl_occ = line_nmbr
        if 'Virtual:' in line:
            vrtl_start = line_nmbr + 1
        if 'Final Occupation by Irrep:' in line:
            vrtl_end = line_nmbr - 1
        line_nmbr += 1

    if rhf:
        dbl_end = vrtl_start - 2
    else:
        dbl_end = sngl_occ - 1
        sngl_start = sngl_occ + 1
        sngl_end = vrtl_start - 2

# Debugging: Print the initial and final line numbers for the various orbital groups
#
#     if not rhf:
#         print(dbl_start, dbl_end, sngl_start, sngl_end, vrtl_start, vrtl_end)
#     else:
#         print(dbl_start, dbl_end, vrtl_start, vrtl_end)

    orb_sym = []

    ## Extract the irreps of doubly occupied orbitals
    p4_output.seek(0)
    for line in islice(p4_output, dbl_start, dbl_end):
        if line.strip():
            x = line.split()
            for i in range(0, len(x), 2):
                try:
                    idx = x[i].index('A')
                except:
                    idx = x[i].index('B')
                orb_sym.append(x[i][idx:].ljust(3))

    ## If rohf reference, extract the irreps of singly occupied orbitals
    if not rhf:
        p4_output.seek(0)
        for line in islice(p4_output, sngl_start, sngl_end):
            if line.strip():
                x = line.split()
                for i in range(0, len(x), 2):
                    try:
                        idx = x[i].index('A')
                    except:
                        idx = x[i].index('B')
                    orb_sym.append(x[i][idx:].ljust(3))

    ## Extract the irreps of doubly occupied orbitals
    p4_output.seek(0)
    for line in islice(p4_output, vrtl_start, vrtl_end):
        if line.strip():
            x = line.split()
            for i in range(0, len(x), 2):
                try:
                    idx = x[i].index('A')
                except:
                    idx = x[i].index('B')
                orb_sym.append(x[i][idx:].ljust(3))

    p4_output.close()

    ## Retrieve the dictionaries of irreps for the given point group
    irrep_to_int, _ = symmetry_irreps(pnt_grp)

    ## Map the irrep of each orbital to the associated integer
    orb_sym_int = list(map(irrep_to_int.get, orb_sym))

    return pnt_grp, orb_sym, orb_sym_int
