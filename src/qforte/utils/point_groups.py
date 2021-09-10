"""
point_groups.py
====================================================
This module contains information about the character
tables of the point groups supported by QForte and
a function that finds the irrep of a given SQOp.
Currently, this module pertains only to molecular
objects with "build_type = 'psi4'".
"""

def char_table(point_group):
    """
    Function that prints the character table of a chosen point group.

    Parameters
    ----------
    point_group: list of two lists; point_group[0] holds the name of the group
                                    point_group[1] is a list that holds the irreps of the group

    """

    if type(point_group) != list:
        raise TypeError("""{0} is not a list.
                This function takes arguments of the form:\n
                [['c2v'], ['A1', 'A2', 'B1', 'B2']]]\n
                using the so-called Cotton ordering of the irreps.""".format(type(point_group)))

    group = point_group[0].lower()

    if group not in ['c1', 'c2', 'ci', 'cs', 'd2', 'c2h', 'c2v', 'd2h']:
        raise ValueError('The given point group is not supported. Choose one of:\nC1, C2, Ci, Cs, D2, C2h, C2v, D2h.')

    irreps = point_group[1]

    print('==========>', group.capitalize(), '<==========')
    print('')
    print('      ', end = '')

    for irrep in irreps:
        print(irrep.ljust(3), ' ', end = '')

    print('\n')

    for idx1, irrep1 in enumerate(irreps):
        print(irrep1.ljust(3), '  ', end = '')
        for idx2, irrep2 in enumerate(irreps):
            print(irreps[idx1 ^ idx2].ljust(3), ' ', end = '')
        print()

def sq_op_find_symmetry(orb_irreps_to_int, annihilate, create):
    """
    Function that finds the irreducible representation of a given
    second-quantized operator.

    Parameters
    ----------
    orb_irreps_to_int: list of integers; each integer corresponds to a particular irrep

    annihilate: list of spinorbital indices to be annihilated

    create: list of spinorbital indices to be created

    """

    sym = 0
    for spinorb in annihilate + create:
        sym ^= orb_irreps_to_int[int(spinorb/2)]

    return sym
