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
from qforte.utils.compact_excitation_circuits import compact_excitation_circuit

class UCC:
    """A mixin class for implementing the UCC circuit ansatz, to be inherited by a
    concrete class UCC+algorithm class.
    """

    def ansatz_circuit(self, amplitudes=None):
        """ This function returns the Circuit object built
        from the appropriate amplitudes.

        Parameters
        ----------
        amplitudes : list
            A list of parameters that define the variational degrees of freedom in
            the state preparation circuit Uvqc. This is needed for the scipy minimizer.
        """
        temp_pool = qf.SQOpPool()
        tamps = self._tamps if amplitudes is None else amplitudes

        if self._compact_excitations:
            U = qf.Circuit()
            for tamp, top in zip(tamps, self._tops):
                U.add(compact_excitation_circuit(tamp * self._pool_obj[top][1].terms()[1][0],
                                                        self._pool_obj[top][1].terms()[1][1],
                                                        self._pool_obj[top][1].terms()[1][2],
                                                        self._qubit_excitations))
            return U

        for tamp, top in zip(tamps, self._tops):
            temp_pool.add(tamp, self._pool_obj[top][1])

        A = temp_pool.get_qubit_operator('commuting_grp_lex', qubit_excitations = self._qubit_excitations)

        U, phase1 = trotterize(A, trotter_number=self._trotter_number)
        if phase1 != 1.0 + 0.0j:
            raise ValueError("Encountered phase change, phase not equal to (1.0 + 0.0i)")
        return U

    def build_orb_energies(self):
        """
        This code provides the spin-orbital energies used in
        Jacobi iterations. If the orbital energies have not
        been provided by an external software (Psi4), they
        are computed internally.
        """
        self._orb_e = []
        ### WARNING: Assuming RHF orbital energies ###
        if hasattr(self._sys, 'hf_orbital_energies'):
            print('\nSingle-particle energies')
            print('------------------------', flush=True)
            for i, j in enumerate(self._sys.hf_orbital_energies):
                self._orb_e += [j]*2
                print(f' {2*i:3} {j:+16.12f}', flush=True)
                print(f' {2*i+1:3} {j:+16.12f}', flush=True)
        else:
            print('\nBuilding single-particle energies:')
            print('---------------------------------------', flush=True)
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

                print(f'  {i:3}     {ei:+16.12f}', flush=True)
                self._orb_e.append(ei)

    def get_res_over_mpdenom(self, residuals):
        """This function returns a vector given by the residuals dividied by the
        respective Moller Plesset denominators.

        Parameters
        ----------
        residuals : list of floats
            The list of (real) floating point numbers which represent the
            residuals.
        """

        resids_over_denoms = []

        # loop over toperators
        for mu, m in enumerate(self._tops):
            sq_op = self._pool_obj[m][1]

            temp_idx = sq_op.terms()[0][2][-1]
            if temp_idx < int(sum(self._ref)/2): # if temp_idx is an occupied idx
                sq_creators = sq_op.terms()[0][1]
                sq_annihilators = sq_op.terms()[0][2]
            else:
                sq_creators = sq_op.terms()[0][2]
                sq_annihilators = sq_op.terms()[0][1]

            denom = sum(self._orb_e[x] for x in sq_annihilators) - sum(self._orb_e[x] for x in sq_creators)

            res_mu = residuals[mu] / denom

            resids_over_denoms.append(res_mu)

        return resids_over_denoms

