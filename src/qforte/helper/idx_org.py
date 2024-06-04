import numpy as np
import qforte as qf


def sorted_largest_idxs(array, use_real=False, rev=True):
    """Sorts the indexes of an array and stores the old indexes.

    Arguments
    ---------

    array : ndarray
        A real (or complex) numpy array to to be sorted.

    use_real : bool
        Whether or not to sort based only on the real portion of the array
        values.

    rev : bool
        Whether to reverse the order of the retruned ndarray.

    Returns
    -------

    sorted_temp : ndarray
        A numpy array of pairs containing the sorted values and the oritional
        index.

    """
    temp = np.empty((len(array)), dtype=object)
    for i, val in enumerate(array):
        temp[i] = (val, i)
    if use_real:
        sorted_temp = sorted(temp, key=lambda factor: np.real(factor[0]), reverse=rev)
    else:
        sorted_temp = sorted(temp, key=lambda factor: factor[0], reverse=rev)
    return sorted_temp


def get_op_from_basis_idx(ref, I):
    max_nbody = 100

    nqb = len(ref)
    nel = int(sum(ref))

    # TODO(Nick): incorparate more flexability into this
    na_el = int(nel / 2)
    nb_el = int(nel / 2)

    basis_I = qf.QubitBasis(I)

    nbody = 0
    pn = 0
    na_I = 0
    nb_I = 0
    holes = []  # i, j, k, ...
    particles = []  # a, b, c, ...
    parity = []

    # for ( p=0; p<nel; p++) {
    for p in range(nel):
        bit_val = int(basis_I.get_bit(p))
        nbody += 1 - bit_val
        pn += bit_val
        if p % 2 == 0:
            na_I += bit_val
        else:
            nb_I += bit_val

        if bit_val - 1:
            holes.append(p)
            if p % 2 == 0:
                parity.append(1)
            else:
                parity.append(-1)

    # for ( q=nel; q<nqb; q++)
    for q in range(nel, nqb):
        bit_val = int(basis_I.get_bit(q))
        pn += bit_val
        if q % 2 == 0:
            na_I += bit_val
        else:
            nb_I += bit_val

        if bit_val:
            particles.append(q)
            if q % 2 == 0:
                parity.append(1)
            else:
                parity.append(-1)

    if pn == nel and na_I == na_el and nb_I == nb_el:
        if nbody != 0 and nbody <= max_nbody:
            total_parity = 1
            # for (const auto& z: parity)
            for z in parity:
                total_parity *= z

            if total_parity == 1:
                # particles.insert(particles.end(), holes.begin(), holes.end());
                excitation = particles + holes
                dexcitation = list(reversed(excitation))
                # std::vector<> particles_adj (particles.rbegin(), particles.rend());
                sigma_I = [1.0, tuple(excitation)]
                # need i, j, a, b
                # SQOperator t_temp;
                # t_temp.add(+1.0, particles);
                # t_temp.add(-1.0, particles_adj);
                # t_temp.simplify();
                # add(1.0, t_temp);
