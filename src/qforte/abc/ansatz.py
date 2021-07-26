"""
Ansatz base classes
====================================
The mixin classes inheritied by any algorithm that uses a parameterized
ansatz. Member functions should be minimal and aim only to implement
the ansatz circut and potential supporting utility functions.
"""

import qforte as qf

from qforte.utils.trotterization import trotterize

class UCC:
    """A mixin class for implementing the UCC circuit ansatz, to be inherited by a
    concrete class UCC+algorithm class.
    """

    def ansatz_circuit(self, amplitudes=None):
        """ This function returns the Circuit object built
        from the appropriate amplitudes (tops).

        Parameters
        ----------
        amplitudes : list
            A list of parameters that define the variational degrees of freedom in
            the state preparation circuit Uvqc. This is needed for the scipy minimizer.
        """
        temp_pool = qf.SQOpPool()
        tamps = self._tamps if amplitudes is None else amplitudes
        for tamp, top in zip(tamps, self._tops):
            temp_pool.add(tamp, self._pool[top][1])

        A = temp_pool.get_qubit_operator('commuting_grp_lex')

        U, phase1 = trotterize(A, trotter_number=self._trotter_number)
        if phase1 != 1.0 + 0.0j:
            raise ValueError("Encountered phase change, phase not equal to (1.0 + 0.0i)")
        return U
