import qforte
import numpy as np


def build_refprep(ref):
    refprep = qforte.Circuit()
    for j, occupied in enumerate(ref):
        if occupied:
            refprep.add(qforte.gate("X", j, j))

    return refprep


def ref_string(ref, nqb):
    temp = ref.copy()
    temp.reverse()
    ref_basis_idx = int("".join(str(x) for x in temp), 2)
    ref_basis = qforte.QubitBasis(ref_basis_idx)
    return ref_basis.str(nqb)


def integer_to_ref(n, nqubits):
    """Takes an integer pertaining to a biary number and returns the corresponding
    determinant occupation list (reference).

        Arguments
        ---------

        n : int
            The index.

        nqubits : int
            The number of qubits (will be length of list).

        Returns
        -------

        ref : list
            A list of 1's and 0's representing spin orbital occupations for a
            single Slater determinant.

    """
    qb = qforte.QubitBasis(n)
    ref = []
    for i in range(nqubits):
        if qb.get_bit(i):
            ref.append(1)
        else:
            ref.append(0)
    return ref


def open_shell(ref):
    """Determines Wheter or not the reference is an open shell determinant.
    Returns True if ref is open shell and False if not.

        Arguments
        ---------

        ref : list
            A list of 1's and 0's representing spin orbital occupations for a
            single Slater determinant.

    """
    norb = int(len(ref) / 2)
    for i in range(norb):
        i_alfa = 2 * i
        i_beta = (2 * i) + 1
        if (ref[i_alfa] + ref[i_beta]) == 1:
            return True

    return False


def correct_spin(ref, abs_spin):
    """Determines Wheter or not the reference has correct spin.
    Returns True if ref is ref spin matches overall spin and False if not.

        Arguments
        ---------

        ref : list
            A list of 1's and 0's representing spin orbital occupations for a
            single Slater determinant.

        abs_spin : float
            The targeted spin value.

    """
    # if (abs_spin != 0):
    #     raise NotImplementedError("MRSQK currently only supports singlet state calculations.")

    norb = int(len(ref) / 2)
    spin = 0.0
    for i in range(norb):
        i_alfa = 2 * i
        i_beta = (2 * i) + 1
        spin += ref[i_alfa] * 0.5
        spin -= ref[i_beta] * 0.5

    if np.abs(spin) == abs_spin:
        return True
    else:
        return False


def flip_spin(ref, orb_idx):
    """Takes in a single determinant reference and returns a determinant with
    the spin of the spin of the specified spatial orbtital (orb_idx) flipped.
    If the specified spatail orbital is doubly occupied, then the same
    determinant is returned.

        Arguments
        ---------

        ref : list
            A list of 1's and 0's representing spin orbital occupations for a
            single Slater determinant.

        orb_idx : int
            An index for the spatial orbtial of interest.

        Retruns
        -------

        temp : list
            A list of 1's and 0's representing spin orbital occupations for a
            single Slater determinant, with the spin of the specified spatial
            orbital flipped.

    """
    temp = ref.copy()
    i_alfa = 2 * orb_idx
    i_beta = (2 * orb_idx) + 1
    alfa_val = ref[i_alfa]
    beta_val = ref[i_beta]

    temp[i_alfa] = beta_val
    temp[i_beta] = alfa_val
    return temp


def build_eq_dets(open_shell_ref):
    """Builds a list of unique spin equivalent determinants from an open shell
    determinant. For example, if [1,0,0,1] is given as in input, it will return
    [[1,0,0,1], [0,1,1,0]].

        Arguments
        ---------

        open_shell_ref : list
            A list of 1's and 0's representing spin orbital occupations for a
            single open-shell Slater determinant.

        Returns
        -------

        eq_ref_lst2 : list of lists
            A list of open-shell determinants which are spin equivalent to
            open_shell_ref (including open_shell_ref).

    """
    norb = int(len(open_shell_ref) / 2)
    one_e_orbs = []
    spin = 0.0
    for i in range(norb):
        i_alfa = 2 * i
        i_beta = (2 * i) + 1
        spin += open_shell_ref[i_alfa] * 0.5
        spin -= open_shell_ref[i_beta] * 0.5

        if (open_shell_ref[i_alfa] + open_shell_ref[i_beta]) == 1:
            one_e_orbs.append(i)

    abs_spin = np.abs(spin)
    eq_ref_lst1 = [open_shell_ref]

    for ref in eq_ref_lst1:
        for orb in one_e_orbs:
            temp = flip_spin(ref, orb)
            if temp not in eq_ref_lst1:
                eq_ref_lst1.append(temp)

    eq_ref_lst2 = []
    for ref in eq_ref_lst1:
        if correct_spin(ref, abs_spin):
            eq_ref_lst2.append(ref)

    return eq_ref_lst2


def ref_to_basis_idx(ref):
    """Turns a reference list into a integer representing its binary value.

    Arguments
    ---------

    ref : list
        The reference determinant (list of 1's and 0's) indicating the spin
        orbtial occupation.

    Returns
    -------

    idx_val : int
        The value of the index.

    """
    temp = ref.copy()
    temp.reverse()
    idx_val = int("".join(str(x) for x in temp), 2)
    return idx_val
