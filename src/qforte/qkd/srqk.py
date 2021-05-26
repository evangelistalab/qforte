"""
srqk.py
=================================================
A module for calculating reference states for quantum
mechanical systems for the multireference selected
quantum Krylov algorithm.
"""

import qforte
from qforte.abc.qsdabc import QSD
from qforte.helper.printing import matprint
# from qforte.utils.transforms import (circuit_to_organizer,
#                                     organizer_to_circuit,
#                                     join_organizers,
#                                     get_jw_organizer)

from qforte.maths.eigsolve import canonical_geig_solve

from qforte.utils.state_prep import *
from qforte.utils.trotterization import (trotterize,
                                         trotterize_w_cRz)

import numpy as np
from scipy.linalg import lstsq

class SRQK(QSD):
    def run(self,
            s=3,
            dt=0.5,
            target_root=0,
            diagonalize_each_step=True
            ):
        """
        Construct a reference state for the MRSQK algorithm as some root of the Hamiltonian in the space
        of H U_n φ where U_m = exp(-i m dt H) and φ a single determinant.

        _diagonalize_each_step : bool
            For diagnostic purposes, should the eigenvalue of the target root of the quantum Krylov subspace
            be printed after each new unitary? We recommend passing an s so the change in the eigenvalue is
            small.
        _dt : float
            The dt used in the unitaries
        _nstates : int
            The number of states
        _s : int
            The greatest m to use in unitaries
        _target_root : int
            Which root of the quantum Krylov subspace should be taken?
        """

        self._s = s
        self._nstates = s+1
        self._dt = dt
        self._target_root = target_root
        self._diagonalize_each_step = diagonalize_each_step

        self._n_classical_params = 0
        self._n_cnot = 0
        self._n_pauli_trm_measures = 0

        # Print options banner (should done for all algorithms).
        self.print_options_banner()

        ######### SRQK #########

        # Build S and H matricies
        if(self._fast):
            self._S, self._Hbar = self.build_qk_mats()
        else:
            self._S, self._Hbar = self.build_qk_mats_realistic()

        # Set the condition number of QSD overlap
        self._Scond = np.linalg.cond(self._S)

        # Get eigenvalues and eigenvectors
        self._eigenvalues, self._eigenvectors \
        = canonical_geig_solve(self._S,
                               self._Hbar,
                               print_mats=self._verbose,
                               sort_ret_vals=True)

        print('\n       ==> QK eigenvalues <==')
        print('----------------------------------------')
        for i, val in enumerate(self._eigenvalues):
            print('  root  {}  {:.8f}    {:.8f}j'.format(i, np.real(val), np.imag(val)))

        # Set ground state energy.
        self._Egs = np.real(self._eigenvalues[0])

        # Set target state energy.
        if(self._target_root==0):
            self._Ets = self._Egs
        else:
            self._Ets = np.real(self._eigenvalues[self._target_root])

        ######### SRQK #########

        # Print summary banner (should done for all algorithms).
        self.print_summary_banner()

        # verify that required attributes were defined
        # (should be called for all algorithms!)
        self.verify_run()

    # Define Algorithm abstract methods.
    def run_realistic(self):
        raise NotImplementedError('run_realistic() is not fully implemented for SRQK.')

    def verify_run(self):
        self.verify_required_attributes()
        self.verify_required_QSD_attributes()

    def print_options_banner(self):
        print('\n-----------------------------------------------------')
        print('           Single Reference Quantum Krylov   ')
        print('-----------------------------------------------------')

        print('\n\n                     ==> QK options <==')
        print('-----------------------------------------------------------')
        # General algorithm options.
        print('Trial reference state:                   ',  ref_string(self._ref, self._nqb))
        print('Number of Hamiltonian Pauli terms:       ',  self._Nl)
        print('Trial state preparation method:          ',  self._state_prep_type)
        print('Trotter order (rho):                     ',  self._trotter_order)
        print('Trotter number (m):                      ',  self._trotter_number)
        print('Use fast version of algorithm:           ',  str(self._fast))
        if(self._fast):
            print('Measurement varience thresh:             ',  'NA')
        else:
            print('Measurement varience thresh:             ',  0.01)

        # Specific SRQK options.
        print('Dimension of Krylov space (N):           ',  self._nstates)
        print('Delta t (in a.u.):                       ',  self._dt)
        print('Target root:                             ',  str(self._target_root))


    def print_summary_banner(self):
        cs_str = '{:.2e}'.format(self._Scond)

        print('\n\n                     ==> QK summary <==')
        print('-----------------------------------------------------------')
        print('Condition number of overlap mat k(S):      ', cs_str)
        print('Final SRQK ground state Energy:           ', round(self._Egs, 10))
        print('Final SRQK target state Energy:           ', round(self._Ets, 10))
        print('Number of classical parameters used:       ', self._n_classical_params)
        print('Number of CNOT gates in deepest circuit:   ', self._n_cnot)
        print('Number of Pauli term measurements:         ', self._n_pauli_trm_measures)

    def build_qk_mats(self):
        """Returns matrices S and H needed for the MRSQK algorithm using the Trotterized
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

        h_mat = np.zeros((self._nstates,self._nstates), dtype=complex)
        s_mat = np.zeros((self._nstates,self._nstates), dtype=complex)

        # Store these vectors for the aid of MRSQK
        self._omega_lst = []
        Homega_lst = []

        if(self._diagonalize_each_step):
            print('\n\n')

            print(f"{'k(S)':>7}{'E(Npar)':>19}{'N(params)':>14}{'N(CNOT)':>18}{'N(measure)':>20}")
            print('-------------------------------------------------------------------------------')

            if (self._print_summary_file):
                f = open("summary.dat", "w+", buffering=1)
                f.write(f"#{'k(S)':>7}{'E(Npar)':>19}{'N(params)':>14}{'N(CNOT)':>18}{'N(measure)':>20}\n")
                f.write('#-------------------------------------------------------------------------------\n')

        for m in range(self._nstates):
            # Compute U_m = exp(-i m dt H)
            Um = qforte.QuantumCircuit()
            Um.add(self._Uprep)
            phase1 = 1.0

            if(m>0):
                fact = (0.0-1.0j) * m * self._dt
                expn_op1, phase1 = trotterize(self._qb_ham, factor=fact, trotter_number=self._trotter_number)
                Um.add(expn_op1)

            # Compute U_m |φ>
            QC = qforte.QuantumComputer(self._nqb)
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

            if (self._diagonalize_each_step):
                # TODO (cleanup): have this print to a separate file
                k = m+1
                evals, evecs = canonical_geig_solve(s_mat[0:k, 0:k],
                                   h_mat[0:k, 0:k],
                                   print_mats=False,
                                   sort_ret_vals=True)

                scond = np.linalg.cond(s_mat[0:k, 0:k])
                self._n_classical_params = k
                self._n_cnot = 2 * Um.get_num_cnots()
                self._n_pauli_trm_measures  = k * self._Nl
                self._n_pauli_trm_measures += k * (k-1) * self._Nl
                self._n_pauli_trm_measures += k * (k-1)

                print(f' {scond:7.2e}    {np.real(evals[self._target_root]):+15.9f}    {self._n_classical_params:8}        {self._n_cnot:10}        {self._n_pauli_trm_measures:12}')
                if (self._print_summary_file):
                    f.write(f'  {scond:7.2e}    {np.real(evals[self._target_root]):+15.9f}    {self._n_classical_params:8}        {self._n_cnot:10}        {self._n_pauli_trm_measures:12}\n')

        if (self._diagonalize_each_step and self._print_summary_file):
            f.close()

        self._n_classical_params = self._nstates
        self._n_cnot = 2 * Um.get_num_cnots()
        # diagonal terms of Hbar
        self._n_pauli_trm_measures  = self._nstates * self._Nl
        # off-diagonal of Hbar (<X> and <Y> of Hadamard test)
        self._n_pauli_trm_measures += self._nstates*(self._nstates-1) * self._Nl
        # off-diagonal of S (<X> and <Y> of Hadamard test)
        self._n_pauli_trm_measures += self._nstates*(self._nstates-1)


        return s_mat, h_mat

    def build_qk_mats_realistic(self):
        h_mat = np.zeros((self._nstates,self._nstates), dtype=complex)
        s_mat = np.zeros((self._nstates,self._nstates), dtype=complex)

        for p in range(self._nstates):
            for q in range(p, self._nstates):
                h_mat[p][q] = self.matrix_element(p, q, use_op=True)
                h_mat[q][p] = np.conj(h_mat[p][q])
                s_mat[p][q] = self.matrix_element(p, q, use_op=False)
                s_mat[q][p] = np.conj(s_mat[p][q])

        return s_mat, h_mat


    def matrix_element(self, m, n, use_op=False):
        """Returns a single matrix element M_mn based on the evolution of
        two unitary operators Um = exp(-i * m * dt * H) and Un = exp(-i * n * dt *H)
        on a reference state |Phi_o>, (optionally) with respect to an operator A.
        Specifically, M_mn is given by <Phi_o| Um^dag Un | Phi_o> or
        (optionally if A is specified) <Phi_o| Um^dag A Un | Phi_o>.

            Arguments
            ---------

            ref : list
                The the reference state |Phi_o>.

            dt : float
                The real time step value (delta t).

            m : int
                The number of time steps for the Um evolution.

            n : int
                The number of time steps for the Un evolution.

            H : QuantumOperator
                The operator to time evolove with respect to (usually the Hamiltonain).

            nqubits : int
                The number of qubits

            A : QuantumOperator
                The overal operator to measure with respect to (optional).

            trot_number : int
                The number of trotter steps (m) to perform when approximating the matrix
                exponentials (Um or Un). For the exponential of two non commuting terms
                e^(A + B), the approximate operator C(m) = (e^(A/m) * e^(B/m))^m is
                exact in the infinite m limit.

            Returns
            -------
            value : complex
                The outcome of measuring <X> and <Y> to determine <2*sigma_+>,
                ultimately the value of the matrix elemet.

        """
        value = 0.0
        ancilla_idx = self._nqb
        Uk = qforte.QuantumCircuit()
        temp_op1 = qforte.QuantumOperator()
        # TODO (opt): move to C side.
        for t in self._qb_ham.terms():
            c, op = t
            phase = -1.0j * n * self._dt * c
            temp_op1.add(phase, op)

        expn_op1, phase1 = trotterize_w_cRz(temp_op1,
                                            ancilla_idx,
                                            trotter_number=self._trotter_number)

        for gate in expn_op1.gates():
            Uk.add(gate)

        Ub = qforte.QuantumCircuit()

        temp_op2 = qforte.QuantumOperator()
        for t in self._qb_ham.terms():
            c, op = t
            phase = -1.0j * m * self._dt * c
            temp_op2.add(phase, op)

        expn_op2, phase2 = trotterize_w_cRz(temp_op2,
                                            ancilla_idx,
                                            trotter_number=self._trotter_number,
                                            Use_open_cRz=False)

        for gate in expn_op2.gates():
            Ub.add(gate)

        if not use_op:
            # TODO (opt): use Uprep
            cir = qforte.QuantumCircuit()
            for j in range(self._nqb):
                if self._ref[j] == 1:
                    cir.add(qforte.gate('X', j, j))

            cir.add(qforte.gate('H', ancilla_idx, ancilla_idx))

            cir.add(Uk)

            cir.add(qforte.gate('X', ancilla_idx, ancilla_idx))
            cir.add(Ub)
            cir.add(qforte.gate('X', ancilla_idx, ancilla_idx))

            X_op = qforte.QuantumOperator()
            x_circ = qforte.QuantumCircuit()
            Y_op = qforte.QuantumOperator()
            y_circ = qforte.QuantumCircuit()

            x_circ.add(qforte.gate('X', ancilla_idx, ancilla_idx))
            y_circ.add(qforte.gate('Y', ancilla_idx, ancilla_idx))

            X_op.add(1.0, x_circ)
            Y_op.add(1.0, y_circ)

            X_exp = qforte.Experiment(self._nqb+1, cir, X_op, 100)
            Y_exp = qforte.Experiment(self._nqb+1, cir, Y_op, 100)

            params = [1.0]
            x_value = X_exp.perfect_experimental_avg(params)
            y_value = Y_exp.perfect_experimental_avg(params)

            value = (x_value + 1.0j * y_value) * phase1 * np.conj(phase2)


        else:
            value = 0.0
            for t in self._qb_ham.terms():
                c, V_l = t

                # TODO (opt):
                cV_l = qforte.QuantumCircuit()
                for gate in V_l.gates():
                    gate_str = gate.gate_id()
                    target = gate.target()
                    control_gate_str = 'c' + gate_str
                    cV_l.add(qforte.gate(control_gate_str, target, ancilla_idx))

                cir = qforte.QuantumCircuit()
                # TODO (opt): use Uprep
                for j in range(self._nqb):
                    if self._ref[j] == 1:
                        cir.add(qforte.gate('X', j, j))

                cir.add(qforte.gate('H', ancilla_idx, ancilla_idx))

                cir.add(Uk)
                cir.add(cV_l)

                cir.add(qforte.gate('X', ancilla_idx, ancilla_idx))
                cir.add(Ub)
                cir.add(qforte.gate('X', ancilla_idx, ancilla_idx))

                X_op = qforte.QuantumOperator()
                x_circ = qforte.QuantumCircuit()
                Y_op = qforte.QuantumOperator()
                y_circ = qforte.QuantumCircuit()

                x_circ.add(qforte.gate('X', ancilla_idx, ancilla_idx))
                y_circ.add(qforte.gate('Y', ancilla_idx, ancilla_idx))

                X_op.add(1.0, x_circ)
                Y_op.add(1.0, y_circ)

                X_exp = qforte.Experiment(self._nqb+1, cir, X_op, 100)
                Y_exp = qforte.Experiment(self._nqb+1, cir, Y_op, 100)

                # TODO (cleanup): Remove params as required arg (Nick)
                params = [1.0]
                x_value = X_exp.perfect_experimental_avg(params)
                y_value = Y_exp.perfect_experimental_avg(params)

                element = (x_value + 1.0j * y_value) * phase1 * np.conj(phase2)
                value += c * element

        return value
