import numpy as np

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
    temp = np.empty((len(array)), dtype=object )
    for i, val in enumerate(array):
        temp[i] = (val, i)
    if(use_real):
        sorted_temp = sorted(temp, key=lambda factor: np.real(factor[0]), reverse=rev)
    else:
        sorted_temp = sorted(temp, key=lambda factor: factor[0], reverse=rev)
    return sorted_temp
