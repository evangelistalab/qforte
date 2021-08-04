"""
Ansatz base classes
====================================
The mixin classes inherited by any algorithm that uses a parameterized
ansatz. Member functions should be minimal and aim only to implement
the ansatz circut and potential supporting utility functions.
"""

import qforte as qf

from qforte.utils.state_prep import build_Uprep
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

    def build_orb_energies(self):
        """Calculates single qubit energies. Used in quasi-Newton updates.
        """
        self._orb_e = []

        print('\nBuilding single particle energies list:')
        print('---------------------------------------')
        qc = qf.Computer(self._nqb)
        qc.apply_circuit(build_Uprep(self._ref, 'occupation_list'))
        E0 = qc.direct_op_exp_val(self._qb_ham)

        for i in range(self._nqb):
            qc = qf.Computer(self._nqb)
            qc.apply_circuit(build_Uprep(self._ref, 'occupation_list'))
            qc.apply_gate(qf.gate('X', i, i))
            Ei = qc.direct_op_exp_val(self._qb_ham)

            if(i<sum(self._ref)):
                ei = E0 - Ei
            else:
                ei = Ei - E0

            print(f'  {i:3}     {ei:+16.12f}')
            self._orb_e.append(ei)
