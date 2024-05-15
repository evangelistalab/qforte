"""
printing.py
====================================================
A module for pretty printing functions for matricies
(usually numpy arrays)and other commonly used data
structures in qforte.
"""


# TODO: Edit format to print only 3 or 4 digits.
def matprint(mat, fmt="g"):
    """Prints (2 X 2) numpy arrays in an intelligable fashion.

    Arguments
    ---------

    mat : ndarray
        A real (or complex) 2 X 2 numpt array to be printed.

    """
    col_maxes = [max([len(("{:" + fmt + "}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
        print("")
