"""
mrsqk.py
=================================================
A class for calculating the energies of quantum-
mechanical systems the multireference selected
quantum Krylov algorithm.
"""

import qforte
from qforte.abc.qsdabc import QSD
from qforte.qkd.srqk import SRQK
from qforte.helper.printing import matprint
from qforte.helper.idx_org import sorted_largest_idxs
from qforte.utils.transforms import (circuit_to_organizer,
                                    organizer_to_circuit,
                                    join_organizers,
                                    get_jw_organizer)

from qforte.maths.eigsolve import canonical_geig_solve

from qforte.utils.state_prep import *
from qforte.utils.trotterization import (trotterize,
                                         trotterize_w_cRz)

import numpy as np
from scipy.linalg import (lstsq,
                          eig)

class MRSQK(QSD):
    def run(self,
            d=2,
            s=3,
            mr_dt=0.5,
            target_root=0,
            reference_generator='SRQK',
            use_phase_based_selection=False,
            use_spin_adapted_refs=True,
            s_o=4,
            dt_o=0.25,
            trotter_order_o=1,
            trotter_number_o=1,
            diagonalize_each_step=True
            ):
        """
        _d : int
            The number of reference states.
        _diagonalize_each_step : bool
            For diagnostic purposes, should the eigenvalue of the target root of the quantum Krylov subspace
            be printed after each new unitary? We recommend passing an s so the change in the eigenvalue is
            small.
        _ninitial_states : bool
        _nstates : int
            The number of states
        _nstates_per_ref : int
            The number of states for a generated reference.
        _reference_generator : {"SRQK"}
            Specifies an algorithm to choose the reference state.
        _s : int
            The greatest m to use in unitaries
        _target_root : int
            Which root of the quantum Krylov subspace should be taken?
        _use_phase_based_selection : bool
        _use_spin_adapted_refs : bool

        SRQK Reference Specific Keywords
        _dt_o : float
            dt for SRQK.
        _s_o : int
            s for SRQK.
        _trotter_number_o : int
            The number of Trotter steps to be used in the SRQK algorithm.
        _trotter_order_o : int
            The operator ordering to be used in the Trotter product.
        """

        self._d = d
        self._s = s
        self._nstates_per_ref = s+1
        self._nstates = d*(s+1)
        self._mr_dt = mr_dt
        self._target_root = target_root

        self._reference_generator = reference_generator
        self._use_phase_based_selection = use_phase_based_selection
        self._use_spin_adapted_refs = use_spin_adapted_refs
        self._s_o = s_o
        self._ninitial_states = s_o + 1
        self._dt_o = dt_o
        self._trotter_order_o = trotter_order_o
        self._trotter_number_o = trotter_number_o

        self._diagonalize_each_step=diagonalize_each_step

        # Print options banner (should done for all algorithms).
        self.print_options_banner()

        ######### MRSQK #########

        # 1. Build the reference wavefunctions.
        if(reference_generator=='SRQK'):
            print('\n  ==> Beginning SRQK for reference selection.')
            self._srqk = SRQK(self._sys,
                              self._ref,
                              trotter_order=self._trotter_order_o,
                              trotter_number=self._trotter_number_o)

            self._srqk.run(s=self._s_o,
                           dt=self._dt_o)

            self._n_classical_params = self._srqk._n_classical_params
            self._n_cnot = self._srqk._n_cnot
            self._n_pauli_trm_measures = self._srqk._n_pauli_trm_measures

            self.build_refs_from_srqk()

            print('\n  ==> SRQK reference selection complete.')

        elif(reference_generator=='ACI'):
            raise NotImplementedError('ACI reference generation not yet available in qforte.')
            print('\n  ==> Beginning ACI for reference selction.')
            print('\n  ==> ACI reference selction complete.')

        else:
            raise ValueError("Incorrect value passed for reference_generator, can be 'SRQK' or 'ACI'.")

        # 2. Build the S and H matrices.
        # Build S and H matricies
        if(self._fast):
            if(self._use_spin_adapted_refs):
                self._S, self._Hbar = self.build_sa_qk_mats()
            else:
                self._S, self._Hbar = self.build_qk_mats()
        else:
            self._S, self._Hbar = self.build_qk_mats_realistic()

        # Set the condition number of QSD overlap
        self._Scond = np.linalg.cond(self._S)

        # 3. Solve the generalized eigenproblem
        # Get eigenvalues and eigenvectors
        self._eigenvalues, self._eigenvectors \
        = canonical_geig_solve(self._S,
                               self._Hbar,
                               print_mats=self._verbose,
                               sort_ret_vals=True)

        # 4. Report and set results.
        print('\n       ==> MRSQK eigenvalues <==')
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

        self._n_classical_params = self._nstates

        # diagonal terms of Hbar
        if(reference_generator=='SRQK'):
            self._n_pauli_trm_measures  = self._nstates * self._Nl + self._srqk._n_pauli_trm_measures
        else:
            raise ValueError('Can only count number of paulit term measurements when using SRQK.')
        # off-diagonal of Hbar (<X> and <Y> of Hadamard test)
        self._n_pauli_trm_measures += self._nstates*(self._nstates-1) * self._Nl
        # off-diagonal of S (<X> and <Y> of Hadamard test)
        self._n_pauli_trm_measures += self._nstates*(self._nstates-1)

        ######### MRSQK #########

        # Print summary banner (should done for all algorithms).
        self.print_summary_banner()

        # verify that required attributes were defined
        # (should be called for all algorithms!)
        self.verify_run()

    # Define Algorithm abstract methods.
    def run_realistic(self):
        raise NotImplementedError('run_realistic() is not fully implemented for MRSQK.')

    def verify_run(self):
        self.verify_required_attributes()
        self.verify_required_QSD_attributes()

    def print_options_banner(self):
        print('\n-----------------------------------------------------')
        print('        Multreference Selected Quantum Krylov   ')
        print('-----------------------------------------------------')

        print('\n\n                 ==> MRSQK options <==')
        print('-----------------------------------------------------------')
        # General algorithm options.
        print('Trial reference state:                   ',  ref_string(self._ref, self._nqb))
        print('Trial state preparation method:          ',  self._trial_state_type)
        print('Trotter order (rho):                     ',  self._trotter_order)
        print('Trotter number (m):                      ',  self._trotter_number)
        print('Use fast version of algorithm:           ',  str(self._fast))
        if(self._fast):
            print('Measurement varience thresh:             ',  'NA')
        else:
            print('Measurement varience thresh:             ',  0.01)

        # Specific QITE options.
        print('Dimension of reference space (d):        ',  self._d)
        print('Time evolutions per reference (s):       ',  self._s)
        print('Dimension of Krylov space (N):           ',  self._d * self._nstates_per_ref)
        print('Delta t (in a.u.):                       ',  self._mr_dt)
        print('Target root:                             ',  str(self._target_root))
        print('Use det. selection with sign:            ',  str(self._use_phase_based_selection))
        print('Use spin adapted references:             ',  str(self._use_spin_adapted_refs))

        print('\n\n     ==> Initial QK options (for ref. selection)  <==')
        print('-----------------------------------------------------------')
        if(self._reference_generator=='SRQK'):
            print('Inital Trotter order (rho_o):            ',  self._trotter_order_o)
            print('Inital Trotter number (m_o):             ',  self._trotter_number_o)
            print('Number of initial time evolutions (s_o): ',  self._s_o)
            print('Dimension of inital Krylov space (N_o):  ',  self._ninitial_states)
            print('Initial delta t_o (in a.u.):             ',  self._dt_o)
            print('\n')

    def print_summary_banner(self):
        cs_str = '{:.2e}'.format(self._Scond)

        print('\n\n                 ==> MRSQK summary <==')
        print('-----------------------------------------------------------')
        print('Condition number of overlap mat k(S):      ', cs_str)
        print('Final MRSQK ground state Energy:          ', round(self._Egs, 10))
        print('Final MRSQK target state Energy:          ', round(self._Ets, 10))
        print('Number of classical parameters used:       ', self._n_classical_params)
        print('Number of CNOT gates in deepest circuit:   ', self._n_cnot)
        print('Number of Pauli term measurements:         ', self._n_pauli_trm_measures)

    # Define QK abstract methods.
    def build_qk_mats(self):
        """Returns matrices P and Q with dimension
        (nstates_per_ref * len(ref_lst) X nstates_per_ref * len(ref_lst))
        based on the evolution of two unitary operators Um = exp(-i * m * dt * H)
        and Un = exp(-i * n * dt *H).

        This is done for all single determinant refrerences |Phi_K> in ref_lst,
        with (Q) and without (P) measuring with respect to the operator H.
        Elements P_mn are given by <Phi_I| Um^dag Un | Phi_J>.
        Elements Q_mn are given by <Phi_I| Um^dag H Un | Phi_J>.
        This function builds P and Q in an efficient manor and gives the same result
        as M built from 'matrix_element', but is unphysical for a quantum computer.

            Arguments
            ---------

            ref_lst : list of lists
                A list containing all of the references |Phi_K> to perfrom evolutions on.

            nstates_per_ref : int
                The number of Krylov basis states to generate for each reference.

            dt_lst : list
                List of time steps to use for each reference (ususally the same for
                all references).

            H : QuantumOperator
                The operator to time evolove and measure with respect to
                (usually the Hamiltonain).

            nqubits : int
                The number of qubits

            trot_number : int
                The number of trotter steps (m) to perform when approximating the matrix
                exponentials (Um or Un). For the exponential of two non commuting terms
                e^(A + B), the approximate operator C(m) = (e^(A/m) * e^(B/m))^m is
                exact in the infinite m limit.

            Returns
            -------
            s_mat : ndarray
                A numpy array containing the elements P_mn

            h_mat : ndarray
                A numpy array containing the elements Q_mn

        """

        num_tot_basis = len(self._single_det_refs) * self._nstates_per_ref

        h_mat = np.zeros((num_tot_basis,num_tot_basis), dtype=complex)
        s_mat = np.zeros((num_tot_basis,num_tot_basis), dtype=complex)

        # TODO (opt): make numpy arrays.
        omega_lst = []
        Homega_lst = []

        for i, ref in enumerate(self._single_det_refs):
            for m in range(self._nstates_per_ref):
                # NOTE: do NOT use Uprep here (is determinant specific).
                Um = qforte.QuantumCircuit()
                for j in range(self._nqb):
                    if ref[j] == 1:
                        Um.add(qforte.gate('X', j, j))
                        phase1 = 1.0

                if(m>0):
                    fact = (0.0-1.0j) * m * self._mr_dt
                    expn_op1, phase1 = trotterize(self._qb_ham, factor=fact, trotter_number=self._trotter_number)
                    Um.add(expn_op1)

                QC = qforte.QuantumComputer(self._nqb)
                QC.apply_circuit(Um)
                QC.apply_constant(phase1)
                omega_lst.append(np.asarray(QC.get_coeff_vec(), dtype=complex))

                QC.apply_operator(self._qb_ham)
                Homega_lst.append(np.asarray(QC.get_coeff_vec(), dtype=complex))

        if(self._diagonalize_each_step):
            print('\n\n')
            print(f"{'k(S)':>7}{'E(Npar)':>19}{'N(params)':>14}{'N(CNOT)':>18}{'N(measure)':>20}")
            print('-------------------------------------------------------------------------------')

        for p in range(num_tot_basis):
            for q in range(p, num_tot_basis):
                h_mat[p][q] = np.vdot(omega_lst[p], Homega_lst[q])
                h_mat[q][p] = np.conj(h_mat[p][q])
                s_mat[p][q] = np.vdot(omega_lst[p], omega_lst[q])
                s_mat[q][p] = np.conj(s_mat[p][q])

            if (self._diagonalize_each_step):
                # TODO (cleanup): have this print to a separate file
                evals, evecs = canonical_geig_solve(s_mat[0:p+1, 0:p+1],
                                   h_mat[0:p+1, 0:p+1],
                                   print_mats=False,
                                   sort_ret_vals=True)

                scond = np.linalg.cond(s_mat[0:p+1, 0:p+1])
                cs_str = '{:.2e}'.format(scond)

                k = p+1
                self._n_classical_params = k
                if(k==1):
                    self._n_cnot = self._srqk._n_cnot
                else:
                    self._n_cnot = 2 * Um.get_num_cnots()
                self._n_pauli_trm_measures  = k * self._Nl + self._srqk._n_pauli_trm_measures
                self._n_pauli_trm_measures += k * (k-1) * self._Nl
                self._n_pauli_trm_measures += k * (k-1)

                print(f' {scond:7.2e}    {np.real(evals[self._target_root]):+15.9f}    {self._n_classical_params:8}        {self._n_cnot:10}        {self._n_pauli_trm_measures:12}')

        return s_mat, h_mat

    def build_classical_CI_mats(self):
        """Builds a classical configuration interaction out of single determinants.
        """
        num_tot_basis = len(self._pre_sa_ref_lst)
        h_CI = np.zeros((num_tot_basis,num_tot_basis), dtype=complex)

        omega_lst = []
        Homega_lst = []

        for i, ref in enumerate(self._pre_sa_ref_lst):
            # NOTE: do NOT use Uprep here (is determinant specific).
            Un = qforte.QuantumCircuit()
            for j in range(self._nqb):
                if ref[j] == 1:
                    Un.add(qforte.gate('X', j, j))

            QC = qforte.QuantumComputer(self._nqb)
            QC.apply_circuit(Un)
            omega_lst.append(np.asarray(QC.get_coeff_vec(), dtype=complex))

            Homega = np.zeros((2**self._nqb), dtype=complex)

            QC.apply_operator(self._qb_ham)
            Homega_lst.append(np.asarray(QC.get_coeff_vec(), dtype=complex))

            for j in range(len(omega_lst)):
                h_CI[i][j] = np.vdot(omega_lst[i], Homega_lst[j])
                h_CI[j][i] = np.conj(h_CI[i][j])

        return h_CI

    def build_sa_qk_mats(self):
        # TODO (cleanup): imporve/update docs
        """Returns matrices P and Q with dimension
        (nstates_per_ref * len(ref_lst) X nstates_per_ref * len(ref_lst))
        based on the evolution of two unitary operators Um = exp(-i * m * dt * H)
        and Un = exp(-i * n * dt *H).

        This is done for all spin adapted refrerences |Phi_K> in ref_lst,
        with (Q) and without (P) measuring with respect to the operator H.
        Elements P_mn are given by <Phi_I| Um^dag Un | Phi_J>.
        Elements Q_mn are given by <Phi_I| Um^dag H Un | Phi_J>.
        This function builds P and Q in an efficient manor and gives the same result
        as M built from 'matrix_element', but is unphysical for a quantum computer.

            Arguments
            ---------

            ref_lst : list of lists
                A list containing all of the spin adapted references |Phi_K> to perfrom evolutions on.
                Is specifically a list of lists of pairs containing coefficient vales
                and a lists pertaning to single determinants.
                As an example,
                ref_lst = [ [ (1.0, [1,1,0,0]) ], [ (0.7071, [0,1,1,0]), (0.7071, [1,0,0,1]) ] ].

            nstates_per_ref : int
                The number of Krylov basis states to generate for each reference.

            dt_lst : list
                List of time steps to use for each reference (ususally the same for
                all references).

            H : QuantumOperator
                The operator to time evolove and measure with respect to
                (usually the Hamiltonain).

            nqubits : int
                The number of qubits

            trot_number : int
                The number of trotter steps (m) to perform when approximating the matrix
                exponentials (Um or Un). For the exponential of two non commuting terms
                e^(A + B), the approximate operator C(m) = (e^(A/m) * e^(B/m))^m is
                exact in the infinite m limit.

            Returns
            -------
            s_mat : ndarray
                A numpy array containing the elements P_mn

            h_mat : ndarray
                A numpy array containing the elements Q_mn

        """

        num_tot_basis = len(self._sa_ref_lst) * self._nstates_per_ref

        h_mat = np.zeros((num_tot_basis,num_tot_basis), dtype=complex)
        s_mat = np.zeros((num_tot_basis,num_tot_basis), dtype=complex)

        omega_lst = []
        Homega_lst = []

        for i, ref in enumerate(self._sa_ref_lst):
            for m in range(self._nstates_per_ref):

                # TODO (cleanup): will need to consider gate count for this part.
                Um = qforte.QuantumCircuit()
                phase1 = 1.0
                if(m>0):
                    fact = (0.0-1.0j) * m * self._mr_dt
                    expn_op1, phase1 = trotterize(self._qb_ham, factor=fact, trotter_number=self._trotter_number)
                    Um.add(expn_op1)

                QC = qforte.QuantumComputer(self._nqb)
                state_prep_lst = []
                for term in ref:
                    coeff = term[0]
                    det = term[1]
                    idx = ref_to_basis_idx(det)
                    state = qforte.QuantumBasis(idx)
                    state_prep_lst.append( (state, coeff) )

                QC.set_state(state_prep_lst)
                QC.apply_circuit(Um)
                QC.apply_constant(phase1)
                omega_lst.append(np.asarray(QC.get_coeff_vec(), dtype=complex))

                QC.apply_operator(self._qb_ham)
                Homega_lst.append(np.asarray(QC.get_coeff_vec(), dtype=complex))

        if(self._diagonalize_each_step):
            print('\n\n')
            print(f"{'k(S)':>7}{'E(Npar)':>19}{'N(params)':>14}{'N(CNOT)':>18}{'N(measure)':>20}")
            print('-------------------------------------------------------------------------------')

        # TODO (opt): add this to previous loop
        for p in range(num_tot_basis):
            for q in range(p, num_tot_basis):
                h_mat[p][q] = np.vdot(omega_lst[p], Homega_lst[q])
                h_mat[q][p] = np.conj(h_mat[p][q])
                s_mat[p][q] = np.vdot(omega_lst[p], omega_lst[q])
                s_mat[q][p] = np.conj(s_mat[p][q])

            if (self._diagonalize_each_step):
                # TODO (cleanup): have this print to a separate file
                evals, evecs = canonical_geig_solve(s_mat[0:p+1, 0:p+1],
                                   h_mat[0:p+1, 0:p+1],
                                   print_mats=False,
                                   sort_ret_vals=True)

                scond = np.linalg.cond(s_mat[0:p+1, 0:p+1])
                cs_str = '{:.2e}'.format(scond)

                k = p+1
                self._n_classical_params = k
                if(k==1):
                    self._n_cnot = self._srqk._n_cnot
                else:
                    self._n_cnot = 2 * Um.get_num_cnots()
                self._n_pauli_trm_measures  = k * self._Nl + self._srqk._n_pauli_trm_measures
                self._n_pauli_trm_measures += k * (k-1) * self._Nl
                self._n_pauli_trm_measures += k * (k-1)

                print(f' {scond:7.2e}    {np.real(evals[self._target_root]):+15.9f}    {self._n_classical_params:8}        {self._n_cnot:10}        {self._n_pauli_trm_measures:12}')

        return s_mat, h_mat

    def build_qk_mats_realistic(self):
        pass

    def build_refs_from_srqk(self):
        # Build a list of single determinant references form partial tomography.
        self.build_refs()
        if (self._use_spin_adapted_refs):
            self.build_sa_refs()

    def get_refs_from_aci(self):
        raise NotImplementedError('ACI reference generation not yet available in qforte.')


    def build_refs(self):
        """Builds a list of single determinant references to be used in the MRSQK
        procedure.

            Arguments
            ---------

            initial_ref : list
                A list of 1's and 0's representing spin orbital occupations for a
                single open-shell Slater determinant. It serves as the reference
                for the initial quantum Krylov calculation used to pick additional
                references.

            d : int
                The totoal number of references to find.

            ninitial_states : int
                The size of the inital quantum Krylov basis to use. Equal to the
                number of inital time evolutions plus one (s_0 +1).

            initial_dt : float
                The time step (delta t) to use in the inital quantum Krylov
                calculation.

            H : QuantumOperator
                The QuantumOperator object to use in MRSQK.

            target_root : int
                Determines which state to return the energy for.

            fast : bool
                Whether or not to use a faster version of the algorithm that bypasses
                measurment (unphysical for quantum computer).

            use_phase_based_selection : bool
                Whether or not to account for sign discrepencaies when selecting important
                determinants from initial quantum Krylov procedure.

            Returns
            -------

            initial_ref_lst : list of lists
                A list of the most important single determinants according to the initial
                quantum Krylov procedure.

        """

        # TODO: re-write this function, it is quite messy

        initial_ref_lst = []
        true_initial_ref_lst = []
        Nis_untruncated = self._ninitial_states

        # Adjust dimension of system in case matrix was ill conditioned.
        if(self._ninitial_states > len(self._srqk._eigenvalues)):
            print('\n', ninitial_states, ' initial states requested, but QK produced ',
                        len(self._srqk._eigenvalues), ' stable roots.\n Using ',
                        len(self._srqk._eigenvalues),
                        'intial states instead.')

            self._ninitial_states = len(self._ninitial_states)

        sorted_evals_idxs = sorted_largest_idxs(self._srqk._eigenvalues, use_real=True, rev=False)
        sorted_evals = np.zeros((self._ninitial_states), dtype=complex)
        sorted_evecs = np.zeros((Nis_untruncated, self._ninitial_states), dtype=complex)
        for n in range(self._ninitial_states):
            old_idx = sorted_evals_idxs[n][1]
            sorted_evals[n]   = self._srqk._eigenvalues[old_idx]
            sorted_evecs[:,n] = self._srqk._eigenvectors[:,old_idx]

        sorted_sq_mod_evecs = sorted_evecs * np.conjugate(sorted_evecs)

        basis_coeff_mat = np.array(self._srqk._omega_lst)
        Cprime = (np.conj(sorted_evecs.transpose())).dot(basis_coeff_mat)
        for n in range(self._ninitial_states):
            for i, val in enumerate(Cprime[n]):
                Cprime[n][i] *= np.conj(val)

        for n in range(self._ninitial_states):
            for i, val in enumerate(basis_coeff_mat[n]):
                basis_coeff_mat[n][i] *= np.conj(val)

        Cprime_sq_mod = (sorted_sq_mod_evecs.transpose()).dot(basis_coeff_mat)

        true_idx_lst = []
        idx_lst = []

        if(self._use_spin_adapted_refs):
            num_single_det_refs = 2*self._d
        else:
            num_single_det_refs = self._d

        if(self._target_root is not None):

            true_sorted_idxs = sorted_largest_idxs(Cprime[self._target_root,:])
            sorted_idxs = sorted_largest_idxs(Cprime_sq_mod[self._target_root,:])

            for n in range(num_single_det_refs):
                idx_lst.append( sorted_idxs[n][1] )
                true_idx_lst.append( true_sorted_idxs[n][1] )

        else:
            raise NotImplementedError("psudo state-avaraged selection approach not yet functional")

        print('\n\n      ==> Initial QK Determinat selection summary  <==')
        print('-----------------------------------------------------------')

        if(self._use_phase_based_selection):
            print('\nMost important determinats:\n')
            print('index                     determinant  ')
            print('----------------------------------------')
            for i, idx in enumerate(true_idx_lst):
                basis = qforte.QuantumBasis(idx)
                print('  ', i+1, '                ', basis.str(self._nqb))

        else:
            print('\nMost important determinats:\n')
            print('index                     determinant  ')
            print('----------------------------------------')
            for i, idx in enumerate(idx_lst):
                basis = qforte.QuantumBasis(idx)
                print('  ', i+1, '                ', basis.str(self._nqb))

        for idx in true_idx_lst:
            true_initial_ref_lst.append(integer_to_ref(idx, self._nqb))

        if(self._ref not in true_initial_ref_lst):
            print('\n***Adding initial referance determinant!***\n')
            for i in range(len(true_initial_ref_lst) - 1):
                true_initial_ref_lst[i+1] = true_initial_ref_lst[i]

            true_initial_ref_lst[0] = initial_ref

        for idx in idx_lst:
            initial_ref_lst.append(integer_to_ref(idx, self._nqb))

        if(self._ref not in initial_ref_lst):
            print('\n***Adding initial referance determinant!***\n')
            staggard_initial_ref_lst = [initial_ref]
            for i in range(len(initial_ref_lst) - 1):
                staggard_initial_ref_lst.append(initial_ref_lst[i].copy())

            initial_ref_lst[0] = initial_ref
            initial_ref_lst = staggard_initial_ref_lst.copy()

        if(self._use_phase_based_selection):
            self._single_det_refs = true_initial_ref_lst

        else:
            self._single_det_refs = initial_ref_lst

    def build_sa_refs(self):
        """Builds a list of spin adapted references to be used in the MRSQK procedure.

        Arguments
        ---------

        initial_ref : list
            A list of 1's and 0's representing spin orbital occupations for a
            single open-shell Slater determinant. It serves as the reference
            for the initial quantum Krylov calculation used to pick additional
            references.

        d : int
            The totoal number of references to find.

        ninitial_states : int
            The size of the inital quantum Krylov basis to use. Equal to the
            number of inital time evolutions plus one (s_0 +1).

        initial_dt : float
            The time step (delta t) to use in the inital quantum Krylov
            calculation.

        H : QuantumOperator
            The QuantumOperator object to use in MRSQK.

        target_root : int
            Determines which state to return the energy for.

        fast : bool
            Whether or not to use a faster version of the algorithm that bypasses
            measurment (unphysical for quantum computer).

        use_phase_based_selection : bool
            Whether or not to account for sign discrepencaies when selecting important
            determinants from initial quantum Krylov procedure.

        Returns
        -------

        sa_ref_lst : list of lists
            A list containing all of the spin adapted references selected in the
            initial quantum Krylov procedure.
            It is specifically a list of lists of pairs containing coefficient vales
            and a lists pertaning to single determinants.
            As an example,
            ref_lst = [ [ (1.0, [1,1,0,0]) ], [ (0.7071, [0,1,1,0]), (0.7071, [1,0,0,1]) ] ].

        """

        if(self._fast==False):
            raise NotImplementedError('Only fast algorithm avalible to build spin adapted refs.')

        target_root = self._target_root
        self._pre_sa_ref_lst = []
        num_refs_per_config = []

        for ref in self._single_det_refs:
            if(ref not in self._pre_sa_ref_lst):
                if(open_shell(ref)):
                    temp = build_eq_dets(ref)
                    self._pre_sa_ref_lst = self._pre_sa_ref_lst + temp
                    num_refs_per_config.append(len(temp))
                else:
                    self._pre_sa_ref_lst.append(ref)
                    num_refs_per_config.append(1)

        h_mat = self.build_classical_CI_mats()

        evals, evecs = eig(h_mat)

        sorted_evals_idxs = sorted_largest_idxs(evals, use_real=True, rev=False)
        sorted_evals = np.zeros((len(evals)), dtype=float)
        sorted_evecs = np.zeros(np.shape(evecs), dtype=float)
        for n in range(len(evals)):
            old_idx = sorted_evals_idxs[n][1]
            sorted_evals[n]   = np.real(evals[old_idx])
            sorted_evecs[:,n] = np.real(evecs[:,old_idx])

        if(np.abs(sorted_evecs[:,0][0]) < 1.0e-6):
            print('Classical CI ground state likely of wrong symmetry, trying other roots!')
            max = len(sorted_evals)
            adjusted_root = 0
            Co_val = 0.0
            while (Co_val < 1.0e-6):
                adjusted_root += 1
                Co_val = np.abs(sorted_evecs[:,adjusted_root][0])

            target_root = adjusted_root
            print('Now using classical CI root: ', target_root)

        target_state = sorted_evecs[:,target_root]
        basis_coeff_lst = []
        norm_basis_coeff_lst = []
        det_lst = []
        coeff_idx = 0
        for num_refs in num_refs_per_config:
            start = coeff_idx
            end = coeff_idx + num_refs

            summ = 0.0
            for val in target_state[start:end]:
                summ += val * val
            temp = [x / np.sqrt(summ) for x in target_state[start:end]]
            norm_basis_coeff_lst.append(temp)

            basis_coeff_lst.append(target_state[start:end])
            det_lst.append(self._pre_sa_ref_lst[start:end])
            coeff_idx += num_refs

        print('\n\n   ==> Classical CI with spin adapted dets summary <==')
        print('-----------------------------------------------------------')
        print('\nList augmented to included all spin \nconfigurations for open shells.')
        print('\n  Coeff                    determinant  ')
        print('----------------------------------------')
        for i, det in enumerate(self._pre_sa_ref_lst):
            qf_det_idx = ref_to_basis_idx(det)
            basis = qforte.QuantumBasis(qf_det_idx)
            if(target_state[i] > 0.0):
                print('   ', round(target_state[i], 4), '                ', basis.str(self._nqb))
            else:
                print('  ', round(target_state[i], 4), '                ', basis.str(self._nqb))

        basis_importnace_lst = []
        for basis_coeff in basis_coeff_lst:
            for coeff in basis_coeff:
                val = 0.0
                val += coeff*coeff
            basis_importnace_lst.append(val)

        sorted_basis_importnace_lst = sorted_largest_idxs(basis_importnace_lst, use_real=True, rev=True)

        # Construct final ref list, of form
        # [ [ (coeff, [1100]) ], [ (coeff, [1001]), (coeff, [0110]) ], ... ]
        print('\n\n        ==> Final MRSQK reference space summary <==')
        print('-----------------------------------------------------------')

        self._sa_ref_lst = []
        for i in range(self._d):
            print('\nRef ', i+1)
            print('---------------------------')
            old_idx = sorted_basis_importnace_lst[i][1]
            basis_vec = []
            for k in range( len(basis_coeff_lst[old_idx]) ):
                temp = ( norm_basis_coeff_lst[old_idx][k], det_lst[old_idx][k] )
                basis_vec.append( temp )

                qf_det_idx = ref_to_basis_idx(temp[1])
                basis = qforte.QuantumBasis(qf_det_idx)
                if(temp[0] > 0.0):
                    print('   ', round(temp[0], 4), '     ', basis.str(self._nqb))
                else:
                    print('  ', round(temp[0], 4), '     ', basis.str(self._nqb))

            self._sa_ref_lst.append(basis_vec)
