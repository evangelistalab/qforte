"""
qforte.py
====================================
The core module of my example project
"""


def about_me(your_name):
    """
    Return the most important thing about a person.

    Parameters
    ----------
    your_name
        A string indicating the name of the person.

    """
    return "The wise {} loves Python.".format(your_name)


def example_doc_function():
    """
    A brief description goes here.

    Parameters
    ----------
    :param n: : type
        A description of what this parameter does
    """


class example_doc_class:
    """
    A class to read, write, and manipulate cube files

    This class assumes that all coordinates (atoms, grid points)
    are stored in atomic units

    Attributes
    ----------
    data : type
        description

    Methods
    -------
    load(filename)
        Load a cube file (standard format)
    save(filename)
        Save a cube file (standard format)
    save_npz(filename)
        Save a cube file using numpy's .npz format
    load_npz(filename)
        Load a cube file using numpy's .npz format
    scale(factor)
        Multiply the data by a given factor
        Performs: self.data *= factor
    add(other)
        To each grid point add the value of grid poins from another CubeFile
        Performs: self.data += other.data
    pointwise_product(other):
        Multiply every grid point with the value of grid points from another CubeFile.
        Performs: self.data *= other.data
    """
