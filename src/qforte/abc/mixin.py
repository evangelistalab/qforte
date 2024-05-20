"""
Lightweight Mixin Classes
====================================
This file contains several mixin classes.
Such classes are intended for multiple inheritance and should always
call super().__init__ in their constructors.
"""


class Trotterizable:
    """
    A mixin class for methods that employ Trotter approximation.

    _trotter_order : int
        The Trotter order to use for exponentiated operators.
        (exact in the infinite limit).

    _trotter_number : int
        The number of trotter steps (m) to perform when approximating the matrix
        exponentials (Um or Un). For the exponential of two non commuting terms
        e^(A + B), the approximate operator C(m) = (e^(A/m) * e^(B/m))^m is
        exact in the infinite m limit.
    """

    def __init__(self, *args, trotter_order=1, trotter_number=1, **kwargs):
        super().__init__(*args, **kwargs)
        self._trotter_order = trotter_order
        self._trotter_number = trotter_number

    def print_trotter_options(self):
        print("Trotter order (rho):                     ", self._trotter_order)
        print("Trotter number (m):                      ", self._trotter_number)
