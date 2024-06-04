"""
SRQK classes
=================================================
Classes for calculating reference states for quantum
mechanical systems for the single referece selected
quantum Krylov algorithm.
"""

import qforte
from qforte.abc.qsdabc import QSD
from qforte.helper.printing import matprint
from qforte.utils.exp_ops import *

from qforte.maths.eigsolve import canonical_geig_solve

from qforte.utils.state_prep import *

import numpy as np


class NTSRQK(QSD):
    """A quantum subspace diagonalization algorithm that generates the many-body
    basis from different durations of non-Trotterized real time evolution:

    .. math::
        | \\Psi_n \\rangle = e^{-i n \\Delta t \\hat{H}} | \\Phi_0 \\rangle

    In practice Trotterization is used to approximate the time evolution operator.

    Attributes
    ----------

    _dt : float
        The time step used in the time evolution unitaries.

    _nstates : int
        The total number of basis states (s + 1).

    _s : int
        The greatest m to use in unitaries, equal to the number of time evolutions.


    """

    def run(self, s=3, dt=0.5, target_root=0, diagonalize_each_step=True):
        if not self._fast:
            raise ValueError(
                "A realistic implementation of non-Trotterized SRQK is unavailable."
            )

        self._s = s
        self._nstates = s + 1
        self._dt = dt
        self._target_root = target_root
        self._diagonalize_each_step = diagonalize_each_step

        self._n_classical_params = 0
        self._n_cnot = 0
        self._n_pauli_trm_measures = 0

        # Print options banner (should done for all algorithms).
        self.print_options_banner()

        self.common_run()

    # Define Algorithm abstract methods.
    def run_realistic(self):
        raise NotImplementedError(
            "run_realistic() is not fully implemented for NTSRQK."
        )

    def verify_run(self):
        self.verify_required_attributes()
        self.verify_required_QSD_attributes()

    def print_options_banner(self):
        print("\n-----------------------------------------------------")
        print("   Non-Trotterized Single Reference Quantum Krylov   ")
        print("-----------------------------------------------------")

        print("\n\n                   ==> NTQK options <==")
        print("-----------------------------------------------------------")

        self.print_generic_options()

        # Specific SRQK options.
        print("Dimension of Krylov space (N):           ", self._nstates)
        print("Delta t (in a.u.):                       ", self._dt)
        print("Target root:                             ", str(self._target_root))

    def print_summary_banner(self):
        cs_str = "{:.2e}".format(self._Scond)

        print("\n\n                     ==> QK summary <==")
        print("-----------------------------------------------------------")
        print("Condition number of overlap mat k(S):      ", cs_str)
        print("Final SRQK ground state Energy:           ", round(self._Egs, 10))
        print("Final SRQK target state Energy:           ", round(self._Ets, 10))
        print("Number of classical parameters used:       ", self._n_classical_params)
        print("Number of CNOT gates in deepest circuit:   ", "N/A")
        print("Number of Pauli term measurements:         ", self._n_pauli_trm_measures)

    def build_qk_mats(self):
        """Returns matrices S and H needed for the QK algorithm using the Trotterized
        form of the unitary operators U_n = exp(-i n dt H)

        The mathematical operations of this function are unphysical for a quantum
        computer, but efficient for a simulator.

        Returns
        -------
        s_mat : ndarray
            A numpy array containing the elements S_mn = <Phi | Um^dag Un | Phi>.
            _nstates by _nstates

        h_mat : ndarray
            A numpy array containing the elements H_mn = <Phi | Um^dag H Un | Phi>
            _nstates by _nstates
        """

        h_mat = np.zeros((self._nstates, self._nstates), dtype=complex)
        s_mat = np.zeros((self._nstates, self._nstates), dtype=complex)

        # Store these vectors for the aid of MRSQK
        self._omega_lst = []
        Homega_lst = []

        if self._diagonalize_each_step:
            print("\n\n")

            print(
                f"{'k(S)':>7}{'E(Npar)':>19}{'N(params)':>14}{'N(CNOT)':>18}{'N(measure)':>20}"
            )
            print(
                "-------------------------------------------------------------------------------"
            )

            if self._print_summary_file:
                f = open("summary.dat", "w+", buffering=1)
                f.write(
                    f"#{'k(S)':>7}{'E(Npar)':>19}{'N(params)':>14}{'N(CNOT)':>18}{'N(measure)':>20}\n"
                )
                f.write(
                    "#-------------------------------------------------------------------------------\n"
                )

        Hsp = get_scipy_csc_from_op(self._qb_ham, -1.0j)
        QC = qforte.Computer(self._nqb)
        QC.apply_circuit(self._Uprep)

        # get the time evolution vectors
        psi_t_vecs = apply_time_evolution_op(QC, Hsp, self._dt, self._nstates)

        for m in range(self._nstates):
            # # Compute U_m |φ>
            QC = qforte.Computer(self._nqb)
            QC.set_coeff_vec(psi_t_vecs[m])
            self._omega_lst.append(np.asarray(QC.get_coeff_vec(), dtype=complex))

            # Compute H U_m |φ>
            QC.apply_operator(self._qb_ham)
            Homega_lst.append(np.asarray(QC.get_coeff_vec(), dtype=complex))

            # Compute S_mn = <φ| U_m^\dagger U_n |φ> and H_mn = <φ| U_m^\dagger H U_n |φ>
            for n in range(len(self._omega_lst)):
                h_mat[m][n] = np.vdot(self._omega_lst[m], Homega_lst[n])
                h_mat[n][m] = np.conj(h_mat[m][n])
                s_mat[m][n] = np.vdot(self._omega_lst[m], self._omega_lst[n])
                s_mat[n][m] = np.conj(s_mat[m][n])

            if self._diagonalize_each_step:
                # TODO (cleanup): have this print to a separate file
                k = m + 1
                evals, evecs = canonical_geig_solve(
                    s_mat[0:k, 0:k],
                    h_mat[0:k, 0:k],
                    print_mats=False,
                    sort_ret_vals=True,
                )

                scond = np.linalg.cond(s_mat[0:k, 0:k])
                self._n_classical_params = k
                self._n_cnot = 0
                self._n_pauli_trm_measures = k * self._Nl
                self._n_pauli_trm_measures += k * (k - 1) * self._Nl
                self._n_pauli_trm_measures += k * (k - 1)

                print(
                    f" {scond:7.2e}    {np.real(evals[self._target_root]):+15.9f}    {self._n_classical_params:8}        {0:10}        {self._n_pauli_trm_measures:12}"
                )
                if self._print_summary_file:
                    f.write(
                        f"  {scond:7.2e}    {np.real(evals[self._target_root]):+15.9f}    {self._n_classical_params:8}        {0:10}        {self._n_pauli_trm_measures:12}\n"
                    )

        if self._diagonalize_each_step and self._print_summary_file:
            f.close()

        self._n_classical_params = self._nstates
        self._n_cnot = 0
        # diagonal terms of Hbar
        self._n_pauli_trm_measures = self._nstates * self._Nl
        # off-diagonal of Hbar (<X> and <Y> of Hadamard test)
        self._n_pauli_trm_measures += self._nstates * (self._nstates - 1) * self._Nl
        # off-diagonal of S (<X> and <Y> of Hadamard test)
        self._n_pauli_trm_measures += self._nstates * (self._nstates - 1)

        return s_mat, h_mat

    # TODO depricate this function
    def matrix_element(self, m, n, use_op=False):
        pass
