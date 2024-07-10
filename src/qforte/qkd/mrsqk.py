"""
MRSQK classes
=================================================
Classes for calculating the energies of quantum-
mechanical systems the multireference selected
quantum Krylov algorithm.
"""

import qforte
from qforte.abc.mixin import Trotterizable
from qforte.abc.qsdabc import QSD
from qforte.qkd.srqk import SRQK
from qforte.helper.printing import matprint
from qforte.helper.idx_org import sorted_largest_idxs
from qforte.utils.transforms import (
    circuit_to_organizer,
    organizer_to_circuit,
    join_organizers,
    get_jw_organizer,
)

from qforte.maths.eigsolve import canonical_geig_solve

from qforte.utils.state_prep import *
from qforte.utils.trotterization import trotterize, trotterize_w_cRz

import numpy as np
from scipy.linalg import eig


class MRSQK(Trotterizable, QSD):
    """A quantum subspace diagonalization algorithm that generates the many-body
    basis from :math:`s` real time evolutions of :math:`d` differnt orthogonal
    reference states :math:`| \\Phi_I \\rangle`:

    .. math::
        | \\Psi_n^I \\rangle = e^{-i n \\Delta t \\hat{H}} | \\Phi_I \\rangle

    In practice Trotterization is used to approximate the time evolution operator.

    The reference states are determined using an initial single-reference QK
    calculation with its own specified values for the time step :math:`\\Delta t_0`
    and number of SRQK time evolutions :math:`s_0`. One can then determine from
    the target SRQK eigenvector :math:`\\vec{C}` an estimate of the most important
    determinats via the importance value

    .. math::
        P_I = \\sum_n^{s_0 + 1} | \\langle \\Phi_I | \\Psi_n \\rangle |^2 |C_n|^2.

    The quantity :math:`| \\langle \\Phi_I | \\Psi_n \\rangle |^2` can be
    approximated by measuring each of time evolved states in the computational
    basis (if the Jordan Wigner transform was used).

    Onece a set of important determinants :math:`\\{ \\Phi_I \\}` is determined it
    is (optionally) augmented to additionally include spin any complements and
    the Hamiltonain is diagonalized in space of the new spin-complement list.
    The final reference states (if _use_spin_adapted_refs==True) are then given
    by the spin adapted (small) linear combinations of determinats with coefficients
    determined by the aformentioned diagonalization.

    Attributes
    ----------

    _d : int
        The number of reference states :math:`\\Phi_I` to use.

    _diagonalize_each_step : bool
        For diagnostic purposes, should the eigenvalue of the target root of the
        quantum Krylov subspace be printed after each new unitary? We recommend
        passing an s such that the change in the eigenvalue is small.

    _nstates : int
        The number MRSQK basis states :math:`d(s+1)`.

    _nstates_per_ref : int
        The number of states for a generated reference :math:`s+1`.

    _reference_generator : {"SRQK"}
        Specifies an algorithm to choose the reference state.

    _s : int
        The number of time evalutions per reference state :math:`s`.

    _target_root : int
        Which root of the quantum Krylov subspace should be taken?

    _use_phase_based_selection : bool
        Whether or not to use the exact phase condition for the
        determination of important references (default is False, as is unphysical
        for quantum hardware).

    _use_spin_adapted_refs : bool
        Whether or not to generate spin-adapted version of the list of imporatnt
        determinants generated from the SRQK calculation.

    _pre_sa_ref_lst : list of lists
        The list of important individual determinants (each specified as an
        occupation number list [1,1,1,0,1,...]) selected during SRQK.

    _sa_ref_lst : list of lists of pairs
        A list containing all of the spin adapted references selected in the
        initial quantum Krylov procedure.
        It is specifically a list of lists of pairs containing coefficient vales
        and a lists pertaning to single determinants.
        As an example:
        ref_lst = [ [ (1.0, [1,1,0,0]) ], [ (0.7071, [0,1,1,0]), (0.7071, [1,0,0,1]) ] ].

    Prelimenary SRQK Reference Specific Keywords

    _dt_o : float
        The time step :math:`\\Delta t` to use for SRQK.

    _s_o : int
        The :math:`s` value for SRQK.

    _ninitial_states : int
        The number of states :math:`(s_0 + 1)` used by the preliminary SRQK
        calcualtion.

    _trotter_number_o : int
        The number of Trotter steps to be used in the SRQK algorithm.

    _trotter_order_o : int
        The operator ordering to be used in the Trotter product.
    """

    def run(
        self,
        d=2,
        s=3,
        mr_dt=0.5,
        target_root=0,
        reference_generator="SRQK",
        use_phase_based_selection=False,
        use_spin_adapted_refs=True,
        s_o=4,
        dt_o=0.25,
        trotter_order_o=1,
        trotter_number_o=1,
        diagonalize_each_step=True,
    ):
        self._d = d
        self._s = s
        self._nstates_per_ref = s + 1
        self._nstates = d * (s + 1)
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

        self._diagonalize_each_step = diagonalize_each_step

        if self._state_prep_type != "occupation_list":
            raise ValueError(
                "MRSQK implementation can only handle occupation_list reference."
            )

        # Print options banner (should done for all algorithms).
        self.print_options_banner()

        ######### MRSQK #########

        # 1. Build the reference wavefunctions.
        if reference_generator == "SRQK":
            print("\n  ==> Beginning SRQK for reference selection.")
            self._srqk = SRQK(
                self._sys,
                self._ref,
                trotter_order=self._trotter_order_o,
                trotter_number=self._trotter_number_o,
            )

            self._srqk.run(s=self._s_o, dt=self._dt_o)

            self._n_classical_params = self._srqk._n_classical_params
            self._n_cnot = self._srqk._n_cnot

            self.build_refs_from_srqk()

            print("\n  ==> SRQK reference selection complete.")

        else:
            raise ValueError(
                "Incorrect value passed for reference_generator, can be 'SRQK'."
            )

        self.common_run()

    # Define Algorithm abstract methods.
    def set_circuit_variables(self):
        self._n_classical_params = self._nstates

        # diagonal terms of Hbar
        if self._reference_generator == "SRQK":
            self._n_pauli_trm_measures = (
                self._nstates * self._Nl + self._srqk._n_pauli_trm_measures
            )
        else:
            raise ValueError(
                "Can only count number of paulit term measurements when using SRQK."
            )
        # off-diagonal of Hbar (<X> and <Y> of Hadamard test)
        self._n_pauli_trm_measures += self._nstates * (self._nstates - 1) * self._Nl
        # off-diagonal of S (<X> and <Y> of Hadamard test)
        self._n_pauli_trm_measures += self._nstates * (self._nstates - 1)

    def run_realistic(self):
        raise NotImplementedError("run_realistic() is not fully implemented for MRSQK.")

    def verify_run(self):
        self.verify_required_attributes()
        self.verify_required_QSD_attributes()

    def print_options_banner(self):
        print("\n-----------------------------------------------------")
        print("        Multreference Selected Quantum Krylov   ")
        print("-----------------------------------------------------")

        print("\n\n                 ==> MRSQK options <==")
        print("-----------------------------------------------------------")

        self.print_generic_options()

        # Specific QITE options.
        print("Dimension of reference space (d):        ", self._d)
        print("Time evolutions per reference (s):       ", self._s)
        print(
            "Dimension of Krylov space (N):           ", self._d * self._nstates_per_ref
        )
        print("Delta t (in a.u.):                       ", self._mr_dt)
        print("Target root:                             ", str(self._target_root))
        print(
            "Use det. selection with sign:            ",
            str(self._use_phase_based_selection),
        )
        print(
            "Use spin adapted references:             ",
            str(self._use_spin_adapted_refs),
        )

        print("\n\n     ==> Initial QK options (for ref. selection)  <==")
        print("-----------------------------------------------------------")
        if self._reference_generator == "SRQK":
            print("Inital Trotter order (rho_o):            ", self._trotter_order_o)
            print("Inital Trotter number (m_o):             ", self._trotter_number_o)
            print("Number of initial time evolutions (s_o): ", self._s_o)
            print("Dimension of inital Krylov space (N_o):  ", self._ninitial_states)
            print("Initial delta t_o (in a.u.):             ", self._dt_o)
            print("\n")

    def print_summary_banner(self):
        cs_str = "{:.2e}".format(self._Scond)

        print("\n\n                 ==> MRSQK summary <==")
        print("-----------------------------------------------------------")
        print("Condition number of overlap mat k(S):      ", cs_str)
        print("Final MRSQK ground state Energy:          ", round(self._Egs, 10))
        print("Final MRSQK target state Energy:          ", round(self._Ets, 10))
        print("Number of classical parameters used:       ", self._n_classical_params)
        print("Number of CNOT gates in deepest circuit:   ", self._n_cnot)
        print("Number of Pauli term measurements:         ", self._n_pauli_trm_measures)

    def build_qk_mats(self):
        if self._use_spin_adapted_refs:
            return self.build_sa_qk_mats()
        else:
            return self.build_qk_mats_fast()

    # Define QK abstract methods.
    def build_qk_mats_fast(self):
        num_tot_basis = len(self._single_det_refs) * self._nstates_per_ref

        h_mat = np.zeros((num_tot_basis, num_tot_basis), dtype=complex)
        s_mat = np.zeros((num_tot_basis, num_tot_basis), dtype=complex)

        # TODO (opt): make numpy arrays.
        omega_lst = []
        Homega_lst = []

        for i, ref in enumerate(self._single_det_refs):
            for m in range(self._nstates_per_ref):
                # NOTE: do NOT use Uprep here (is determinant specific).
                Um = qforte.Circuit()
                for j in range(self._nqb):
                    if ref[j] == 1:
                        Um.add(qforte.gate("X", j, j))
                        phase1 = 1.0

                if m > 0:
                    fact = (0.0 - 1.0j) * m * self._mr_dt
                    expn_op1, phase1 = trotterize(
                        self._qb_ham, factor=fact, trotter_number=self._trotter_number
                    )
                    Um.add(expn_op1)

                QC = qforte.Computer(self._nqb)
                QC.apply_circuit(Um)
                QC.apply_constant(phase1)
                omega_lst.append(np.asarray(QC.get_coeff_vec(), dtype=complex))

                QC.apply_operator(self._qb_ham)
                Homega_lst.append(np.asarray(QC.get_coeff_vec(), dtype=complex))

        if self._diagonalize_each_step:
            print("\n\n")
            print(
                f"{'k(S)':>7}{'E(Npar)':>19}{'N(params)':>14}{'N(CNOT)':>18}{'N(measure)':>20}"
            )
            print(
                "-------------------------------------------------------------------------------"
            )

        for p in range(num_tot_basis):
            for q in range(p, num_tot_basis):
                h_mat[p][q] = np.vdot(omega_lst[p], Homega_lst[q])
                h_mat[q][p] = np.conj(h_mat[p][q])
                s_mat[p][q] = np.vdot(omega_lst[p], omega_lst[q])
                s_mat[q][p] = np.conj(s_mat[p][q])

            if self._diagonalize_each_step:
                # TODO (cleanup): have this print to a separate file
                evals, evecs = canonical_geig_solve(
                    s_mat[0 : p + 1, 0 : p + 1],
                    h_mat[0 : p + 1, 0 : p + 1],
                    print_mats=False,
                    sort_ret_vals=True,
                )

                scond = np.linalg.cond(s_mat[0 : p + 1, 0 : p + 1])
                cs_str = "{:.2e}".format(scond)

                k = p + 1
                self._n_classical_params = k
                if k == 1:
                    self._n_cnot = self._srqk._n_cnot
                else:
                    self._n_cnot = 2 * Um.get_num_cnots()
                self._n_pauli_trm_measures = (
                    k * self._Nl + self._srqk._n_pauli_trm_measures
                )
                self._n_pauli_trm_measures += k * (k - 1) * self._Nl
                self._n_pauli_trm_measures += k * (k - 1)

                print(
                    f" {scond:7.2e}    {np.real(evals[self._target_root]):+15.9f}    {self._n_classical_params:8}        {self._n_cnot:10}        {self._n_pauli_trm_measures:12}"
                )

        return s_mat, h_mat

    def build_classical_CI_mats(self):
        """Builds a classical configuration interaction out of single determinants."""
        num_tot_basis = len(self._pre_sa_ref_lst)
        h_CI = np.zeros((num_tot_basis, num_tot_basis), dtype=complex)

        omega_lst = []
        Homega_lst = []

        for i, ref in enumerate(self._pre_sa_ref_lst):
            # NOTE: do NOT use Uprep here (is determinant specific).
            Un = qforte.Circuit()
            for j in range(self._nqb):
                if ref[j] == 1:
                    Un.add(qforte.gate("X", j, j))

            QC = qforte.Computer(self._nqb)
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
        """Returns the QK effective hamiltonain and overlap matrices in a basis
        of spin adapted references.
        """

        num_tot_basis = len(self._sa_ref_lst) * self._nstates_per_ref

        h_mat = np.zeros((num_tot_basis, num_tot_basis), dtype=complex)
        s_mat = np.zeros((num_tot_basis, num_tot_basis), dtype=complex)

        omega_lst = []
        Homega_lst = []

        for i, ref in enumerate(self._sa_ref_lst):
            for m in range(self._nstates_per_ref):
                Um = qforte.Circuit()
                phase1 = 1.0
                if m > 0:
                    fact = (0.0 - 1.0j) * m * self._mr_dt
                    expn_op1, phase1 = trotterize(
                        self._qb_ham, factor=fact, trotter_number=self._trotter_number
                    )
                    Um.add(expn_op1)

                QC = qforte.Computer(self._nqb)
                state_prep_lst = []
                for term in ref:
                    coeff = term[0]
                    det = term[1]
                    idx = ref_to_basis_idx(det)
                    state = qforte.QubitBasis(idx)
                    state_prep_lst.append((state, coeff))

                QC.set_state(state_prep_lst)
                QC.apply_circuit(Um)
                QC.apply_constant(phase1)
                omega_lst.append(np.asarray(QC.get_coeff_vec(), dtype=complex))

                QC.apply_operator(self._qb_ham)
                Homega_lst.append(np.asarray(QC.get_coeff_vec(), dtype=complex))

        if self._diagonalize_each_step:
            print("\n\n")
            print(
                f"{'k(S)':>7}{'E(Npar)':>19}{'N(params)':>14}{'N(CNOT)':>18}{'N(measure)':>20}"
            )
            print(
                "-------------------------------------------------------------------------------"
            )

        # TODO (opt): add this to previous loop
        for p in range(num_tot_basis):
            for q in range(p, num_tot_basis):
                h_mat[p][q] = np.vdot(omega_lst[p], Homega_lst[q])
                h_mat[q][p] = np.conj(h_mat[p][q])
                s_mat[p][q] = np.vdot(omega_lst[p], omega_lst[q])
                s_mat[q][p] = np.conj(s_mat[p][q])

            if self._diagonalize_each_step:
                # TODO (cleanup): have this print to a separate file
                evals, evecs = canonical_geig_solve(
                    s_mat[0 : p + 1, 0 : p + 1],
                    h_mat[0 : p + 1, 0 : p + 1],
                    print_mats=False,
                    sort_ret_vals=True,
                )

                scond = np.linalg.cond(s_mat[0 : p + 1, 0 : p + 1])
                cs_str = "{:.2e}".format(scond)

                k = p + 1
                self._n_classical_params = k
                if k == 1:
                    self._n_cnot = self._srqk._n_cnot
                else:
                    self._n_cnot = 2 * Um.get_num_cnots()
                self._n_pauli_trm_measures = (
                    k * self._Nl + self._srqk._n_pauli_trm_measures
                )
                self._n_pauli_trm_measures += k * (k - 1) * self._Nl
                self._n_pauli_trm_measures += k * (k - 1)

                print(
                    f" {scond:7.2e}    {np.real(evals[self._target_root]):+15.9f}    {self._n_classical_params:8}        {self._n_cnot:10}        {self._n_pauli_trm_measures:12}"
                )

        return s_mat, h_mat

    def build_refs_from_srqk(self):
        self.build_refs()
        if self._use_spin_adapted_refs:
            self.build_sa_refs()

    def get_refs_from_aci(self):
        raise NotImplementedError(
            "ACI reference generation not yet available in qforte."
        )

    def build_refs(self):
        """Builds a list of single determinant references (non spin-adapted) to
        be used in the MRSQK procedure.
        """

        initial_ref_lst = []
        true_initial_ref_lst = []
        Nis_untruncated = self._ninitial_states

        # Adjust dimension of system in case matrix was ill conditioned.
        if self._ninitial_states > len(self._srqk._eigenvalues):
            print(
                "\n",
                self._ninitial_states,
                " initial states requested, but QK produced ",
                len(self._srqk._eigenvalues),
                " stable roots.\n Using ",
                len(self._srqk._eigenvalues),
                "intial states instead.",
            )

            self._ninitial_states = len(self._srqk._eigenvalues)

        sorted_evals_idxs = sorted_largest_idxs(
            self._srqk._eigenvalues, use_real=True, rev=False
        )
        sorted_evals = np.zeros((self._ninitial_states), dtype=complex)
        sorted_evecs = np.zeros((Nis_untruncated, self._ninitial_states), dtype=complex)
        for n in range(self._ninitial_states):
            old_idx = sorted_evals_idxs[n][1]
            sorted_evals[n] = self._srqk._eigenvalues[old_idx]
            sorted_evecs[:, n] = self._srqk._eigenvectors[:, old_idx]

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

        if self._use_spin_adapted_refs:
            num_single_det_refs = 2 * self._d
        else:
            num_single_det_refs = self._d

        if self._target_root is not None:
            true_sorted_idxs = sorted_largest_idxs(Cprime[self._target_root, :])
            sorted_idxs = sorted_largest_idxs(Cprime_sq_mod[self._target_root, :])

            for n in range(num_single_det_refs):
                idx_lst.append(sorted_idxs[n][1])
                true_idx_lst.append(true_sorted_idxs[n][1])

        else:
            raise NotImplementedError(
                "psudo state-avaraged selection approach not yet functional"
            )

        print("\n\n      ==> Initial QK Determinat selection summary  <==")
        print("-----------------------------------------------------------")

        if self._use_phase_based_selection:
            print("\nMost important determinats:\n")
            print("index                     determinant  ")
            print("----------------------------------------")
            for i, idx in enumerate(true_idx_lst):
                basis = qforte.QubitBasis(idx)
                print("  ", i + 1, "                ", basis.str(self._nqb))

        else:
            print("\nMost important determinats:\n")
            print("index                     determinant  ")
            print("----------------------------------------")
            for i, idx in enumerate(idx_lst):
                basis = qforte.QubitBasis(idx)
                print("  ", i + 1, "                ", basis.str(self._nqb))

        for idx in true_idx_lst:
            true_initial_ref_lst.append(integer_to_ref(idx, self._nqb))

        if self._ref not in true_initial_ref_lst:
            print("\n***Adding initial referance determinant!***\n")
            for i in range(len(true_initial_ref_lst) - 1):
                true_initial_ref_lst[i + 1] = true_initial_ref_lst[i]

            true_initial_ref_lst[0] = initial_ref

        for idx in idx_lst:
            initial_ref_lst.append(integer_to_ref(idx, self._nqb))

        if self._ref not in initial_ref_lst:
            print("\n***Adding initial referance determinant!***\n")
            staggard_initial_ref_lst = [initial_ref]
            for i in range(len(initial_ref_lst) - 1):
                staggard_initial_ref_lst.append(initial_ref_lst[i].copy())

            initial_ref_lst[0] = initial_ref
            initial_ref_lst = staggard_initial_ref_lst.copy()

        if self._use_phase_based_selection:
            self._single_det_refs = true_initial_ref_lst

        else:
            self._single_det_refs = initial_ref_lst

    def build_sa_refs(self):
        """Builds a list of spin adapted references to be used in the MRSQK procedure."""

        if self._fast == False:
            raise NotImplementedError(
                "Only fast algorithm avalible to build spin adapted refs."
            )

        target_root = self._target_root
        self._pre_sa_ref_lst = []
        num_refs_per_config = []

        for ref in self._single_det_refs:
            if ref not in self._pre_sa_ref_lst:
                if open_shell(ref):
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
            sorted_evals[n] = np.real(evals[old_idx])
            sorted_evecs[:, n] = np.real(evecs[:, old_idx])

        if np.abs(sorted_evecs[:, 0][0]) < 1.0e-6:
            print(
                "Classical CI ground state likely of wrong symmetry, trying other roots!"
            )
            max = len(sorted_evals)
            adjusted_root = 0
            Co_val = 0.0
            while Co_val < 1.0e-6:
                adjusted_root += 1
                Co_val = np.abs(sorted_evecs[:, adjusted_root][0])

            target_root = adjusted_root
            print("Now using classical CI root: ", target_root)

        target_state = sorted_evecs[:, target_root]
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

        print("\n\n   ==> Classical CI with spin adapted dets summary <==")
        print("-----------------------------------------------------------")
        print("\nList augmented to included all spin \nconfigurations for open shells.")
        print("\n  Coeff                    determinant  ")
        print("----------------------------------------")
        for i, det in enumerate(self._pre_sa_ref_lst):
            qf_det_idx = ref_to_basis_idx(det)
            basis = qforte.QubitBasis(qf_det_idx)
            if target_state[i] > 0.0:
                print(
                    "   ",
                    round(target_state[i], 4),
                    "                ",
                    basis.str(self._nqb),
                )
            else:
                print(
                    "  ",
                    round(target_state[i], 4),
                    "                ",
                    basis.str(self._nqb),
                )

        basis_importnace_lst = []
        for basis_coeff in basis_coeff_lst:
            for coeff in basis_coeff:
                val = 0.0
                val += coeff * coeff
            basis_importnace_lst.append(val)

        sorted_basis_importnace_lst = sorted_largest_idxs(
            basis_importnace_lst, use_real=True, rev=True
        )

        print("\n\n        ==> Final MRSQK reference space summary <==")
        print("-----------------------------------------------------------")

        self._sa_ref_lst = []
        for i in range(self._d):
            print("\nRef ", i + 1)
            print("---------------------------")
            old_idx = sorted_basis_importnace_lst[i][1]
            basis_vec = []
            for k in range(len(basis_coeff_lst[old_idx])):
                temp = (norm_basis_coeff_lst[old_idx][k], det_lst[old_idx][k])
                basis_vec.append(temp)

                qf_det_idx = ref_to_basis_idx(temp[1])
                basis = qforte.QubitBasis(qf_det_idx)
                if temp[0] > 0.0:
                    print("   ", round(temp[0], 4), "     ", basis.str(self._nqb))
                else:
                    print("  ", round(temp[0], 4), "     ", basis.str(self._nqb))

            self._sa_ref_lst.append(basis_vec)
