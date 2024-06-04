"""
SRQK classes
=================================================
Classes for calculating reference states for quantum
mechanical systems for the single referece selected
quantum Krylov algorithm.
"""

import qforte
from qforte.abc.mixin import Trotterizable
from qforte.abc.qsdabc import QSD
from qforte.helper.printing import matprint

from qforte.maths.eigsolve import canonical_geig_solve

from qforte.utils.state_prep import *
from qforte.utils.trotterization import trotterize, trotterize_w_cRz

import numpy as np


class SRQK(Trotterizable, QSD):
    """A quantum subspace diagonalization algorithm that generates the many-body
    basis from different durations of real time evolution:

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
        raise NotImplementedError("run_realistic() is not fully implemented for SRQK.")

    def verify_run(self):
        self.verify_required_attributes()
        self.verify_required_QSD_attributes()

    def print_options_banner(self):
        print("\n-----------------------------------------------------")
        print("           Single Reference Quantum Krylov   ")
        print("-----------------------------------------------------")

        print("\n\n                     ==> QK options <==")
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
        print("Number of CNOT gates in deepest circuit:   ", self._n_cnot)
        print("Number of Pauli term measurements:         ", self._n_pauli_trm_measures)

    def build_qk_mats(self):
        if self._fast:
            return self.build_qk_mats_fast()
        else:
            return self.build_qk_mats_realistic()

    def build_qk_mats_fast(self):
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
            string = f"{'k(S)':>7}{'E(Npar)':>19}{'N(params)':>14}{'N(CNOT)':>18}{'N(measure)':>20}\n"
            string += "-------------------------------------------------------------------------------"
            print(string)

            if self._print_summary_file:
                f = open("summary.dat", "w+", buffering=1)
                f.write(string + "\n")

        for m in range(self._nstates):
            # Compute U_m = exp(-i m dt H)
            Um = qforte.Circuit(self._Uprep)
            phase1 = 1.0

            if m > 0:
                fact = (0.0 - 1.0j) * m * self._dt
                expn_op1, phase1 = trotterize(
                    self._qb_ham, factor=fact, trotter_number=self._trotter_number
                )
                Um.add(expn_op1)

            # Compute U_m |φ>
            QC = qforte.Computer(self._nqb)
            QC.apply_circuit(Um)
            QC.apply_constant(phase1)
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
                self._n_cnot = 2 * Um.get_num_cnots()
                self._n_pauli_trm_measures = k * self._Nl
                self._n_pauli_trm_measures += k * (k - 1) * self._Nl
                self._n_pauli_trm_measures += k * (k - 1)

                print(
                    f" {scond:7.2e}    {np.real(evals[self._target_root]):+15.9f}    {self._n_classical_params:8}        {self._n_cnot:10}        {self._n_pauli_trm_measures:12}"
                )
                if self._print_summary_file:
                    f.write(
                        f"  {scond:7.2e}    {np.real(evals[self._target_root]):+15.9f}    {self._n_classical_params:8}        {self._n_cnot:10}        {self._n_pauli_trm_measures:12}\n"
                    )

        if self._diagonalize_each_step and self._print_summary_file:
            f.close()

        self._n_classical_params = self._nstates
        self._n_cnot = 2 * Um.get_num_cnots()
        # diagonal terms of Hbar
        self._n_pauli_trm_measures = self._nstates * self._Nl
        # off-diagonal of Hbar (<X> and <Y> of Hadamard test)
        self._n_pauli_trm_measures += self._nstates * (self._nstates - 1) * self._Nl
        # off-diagonal of S (<X> and <Y> of Hadamard test)
        self._n_pauli_trm_measures += self._nstates * (self._nstates - 1)

        return s_mat, h_mat

    def build_qk_mats_realistic(self):
        h_mat = np.zeros((self._nstates, self._nstates), dtype=complex)
        s_mat = np.zeros((self._nstates, self._nstates), dtype=complex)

        for p in range(self._nstates):
            for q in range(p, self._nstates):
                h_mat[p][q] = self.matrix_element(p, q, use_op=True)
                h_mat[q][p] = np.conj(h_mat[p][q])
                s_mat[p][q] = self.matrix_element(p, q, use_op=False)
                s_mat[q][p] = np.conj(s_mat[p][q])

        return s_mat, h_mat

    # TODO deprecate this function
    def matrix_element(self, m, n, use_op=False):
        """Returns a single matrix element M_mn based on the evolution of
        two unitary operators Um = exp(-i * m * dt * H) and Un = exp(-i * n * dt * H)
        on a reference state |Phi_o>, (optionally) with respect to an operator A.
        Specifically, M_mn is given by <Phi_o| Um^dag Un | Phi_o> or
        (optionally if A is specified) <Phi_o| Um^dag A Un | Phi_o>.

        Arguments
        ---------

        m : int
            The number of time steps for the Um evolution.

        n : int
            The number of time steps for the Un evolution.

        use_op : bool
            Should the matrix element be of H (true) or S (false)?

        Returns
        -------
        value : complex
            The outcome of measuring <X> and <Y> to determine <2*sigma_+>,
            ultimately the value of the matrix elemet.

        """
        value = 0.0
        ancilla_idx = self._nqb
        Uk = qforte.Circuit()
        temp_op1 = qforte.QubitOperator()
        # TODO (opt): move to C side.
        for t in self._qb_ham.terms():
            c, op = t
            phase = -1.0j * n * self._dt * c
            temp_op1.add(phase, op)

        expn_op1, phase1 = trotterize_w_cRz(
            temp_op1, ancilla_idx, trotter_number=self._trotter_number
        )

        for gate in expn_op1.gates():
            Uk.add(gate)

        Ub = qforte.Circuit()

        temp_op2 = qforte.QubitOperator()
        for t in self._qb_ham.terms():
            c, op = t
            phase = -1.0j * m * self._dt * c
            temp_op2.add(phase, op)

        expn_op2, phase2 = trotterize_w_cRz(
            temp_op2,
            ancilla_idx,
            trotter_number=self._trotter_number,
            Use_open_cRz=False,
        )

        for gate in expn_op2.gates():
            Ub.add(gate)

        if not use_op:
            cir = qforte.Circuit(self._Uprep)

            cir.add(qforte.gate("H", ancilla_idx, ancilla_idx))

            cir.add(Uk)

            cir.add(qforte.gate("X", ancilla_idx, ancilla_idx))
            cir.add(Ub)
            cir.add(qforte.gate("X", ancilla_idx, ancilla_idx))

            X_op = qforte.QubitOperator()
            x_circ = qforte.Circuit()
            Y_op = qforte.QubitOperator()
            y_circ = qforte.Circuit()

            x_circ.add(qforte.gate("X", ancilla_idx, ancilla_idx))
            y_circ.add(qforte.gate("Y", ancilla_idx, ancilla_idx))

            X_op.add(1.0, x_circ)
            Y_op.add(1.0, y_circ)

            X_exp = qforte.Experiment(self._nqb + 1, cir, X_op, 100)
            Y_exp = qforte.Experiment(self._nqb + 1, cir, Y_op, 100)

            x_value = X_exp.perfect_experimental_avg()
            y_value = Y_exp.perfect_experimental_avg()

            value = (x_value + 1.0j * y_value) * phase1 * np.conj(phase2)

        else:
            value = 0.0
            for t in self._qb_ham.terms():
                c, V_l = t

                # TODO (opt):
                cV_l = qforte.Circuit()
                for gate in V_l.gates():
                    gate_str = gate.gate_id()
                    target = gate.target()
                    control_gate_str = "c" + gate_str
                    cV_l.add(qforte.gate(control_gate_str, target, ancilla_idx))

                cir = qforte.Circuit(self._Uprep)

                cir.add(qforte.gate("H", ancilla_idx, ancilla_idx))

                cir.add(Uk)
                cir.add(cV_l)

                cir.add(qforte.gate("X", ancilla_idx, ancilla_idx))
                cir.add(Ub)
                cir.add(qforte.gate("X", ancilla_idx, ancilla_idx))

                X_op = qforte.QubitOperator()
                x_circ = qforte.Circuit()
                Y_op = qforte.QubitOperator()
                y_circ = qforte.Circuit()

                x_circ.add(qforte.gate("X", ancilla_idx, ancilla_idx))
                y_circ.add(qforte.gate("Y", ancilla_idx, ancilla_idx))

                X_op.add(1.0, x_circ)
                Y_op.add(1.0, y_circ)

                X_exp = qforte.Experiment(self._nqb + 1, cir, X_op, 100)
                Y_exp = qforte.Experiment(self._nqb + 1, cir, Y_op, 100)

                x_value = X_exp.perfect_experimental_avg()
                y_value = Y_exp.perfect_experimental_avg()

                element = (x_value + 1.0j * y_value) * phase1 * np.conj(phase2)
                value += c * element

        return value
