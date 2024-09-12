"""
UCC-VQE base classes
====================================
The abstract base classes inheritied by any variational quantum eigensolver (VQE)
variant that utilizes a unitary coupled cluster (UCC) type ansatz.
"""

import qforte as qf
from abc import abstractmethod
from qforte.abc.vqeabc import VQE
from qforte.abc.ansatz import UCC

from qforte.experiment import *
from qforte.utils.transforms import *
from qforte.utils.state_prep import ref_to_basis_idx
from qforte.utils.trotterization import trotterize
from qforte.utils.compact_excitation_circuits import compact_excitation_circuit

import numpy as np


class UCCVQE(UCC, VQE):
    """The abstract base class inheritied by any algorithm that seeks to find
    eigenstates by variational minimization of the Energy

    .. math::
        E(\\mathbf{t}) = \\langle \\Phi_0 | \\hat{U}^\\dagger(\\mathbf{\\mathbf{t}}) \\hat{H} \\hat{U}(\\mathbf{\\mathbf{t}}) | \\Phi_0 \\rangle

    using a disentangled UCC type ansatz

    .. math::
        \\hat{U}(\\mathbf{t}) = \\prod_\\mu e^{t_\\mu (\\hat{\\tau}_\\mu - \\hat{\\tau}_\\mu^\\dagger)},

    were :math:`\\hat{\\tau}_\\mu` is a Fermionic excitation operator and
    :math:`t_\\mu` is a cluster amplitude.

    Attributes
    ----------

    _pool_type : string or SQOpPool
        Specifies the kinds of tamplitudes allowed in the UCCN-VQE
        parameterization. If an SQOpPool is supplied, that is used as the
        operator pool. The following strings are allowed:
            SA_SD: At most two orbital excitations. Assumes a singlet wavefunction and closed-shell Slater determinant
                   reducing the number of amplitudes.
            SD: At most two orbital excitations.
            SDT: At most three orbital excitations.
            SDTQ: At most four orbital excitations.
            SDTQP: At most five orbital excitations.
            SDTQPH: At most six orbital excitations.
            GSD: At most two excitations, from any orbital to any orbital.

    _prev_energy : float
        The energy from the previous iteration.

    _curr_energy : float
        The energy from the current iteration.

    _curr_grad_norm : float
        The current norm of the gradient

    _Nm : int
        A list containing the number of pauli terms in each Jordan-Wigner
        transformed excitaiton/de-excitaion operator in the pool.

    _use_analytic_grad : bool
        Whether or not to use an analytic function for the gradient to pass to
        the optimizer. If false, the optimizer will use self-generated approximate
        gradients from finite differences (if BFGS algorithm is used).

    """

    def __init__(self, *args, **kwargs):
        self.computer_initializable = True
        super().__init__(*args, **kwargs)

    @abstractmethod
    def get_num_ham_measurements(self):
        pass

    @abstractmethod
    def get_num_commut_measurements(self):
        pass

    def fill_commutator_pool(self):
        print("\n\n==> Building commutator pool for gradient measurement.")
        self._commutator_pool = self._pool_obj.get_qubit_op_pool()
        self._commutator_pool.join_as_commutator(self._qb_ham)
        print("==> Commutator pool construction complete.")

    def measure_operators(self, operators, Ucirc, idxs=[]):
        """
        Parameters
        ----------
        operators : QubitOpPool
            All operators to be measured

        Ucirc : Circuit
            The state preparation circuit.

        idxs : list of int
            The indices of select operators in the pool of operators. If provided, only these
            operators will be measured.

        """

        if self._fast:
            myQC = self.get_initial_computer()
            myQC.apply_circuit(Ucirc)
            if not idxs:
                grads = myQC.direct_oppl_exp_val(operators)
            else:
                grads = myQC.direct_idxd_oppl_exp_val(operators, idxs)

        else:
            raise NotImplementedError("Must have self._fast to measure an operator.")

        np.testing.assert_allclose(np.imag(grads), np.zeros_like(grads), atol=1e-7)

        return np.real(grads)

    def measure_gradient(self, params=None):
        """Returns the disentangled (factorized) UCC gradient, using a
        recursive approach, as described in Section D of the Appendix of
        10.1038/s41467-019-10988-2

        Parameters
        ----------
        params : list of floats
            The variational parameters which characterize _Uvqc.
        """

        if not self._fast:
            raise ValueError("self._fast must be True for gradient measurement.")

        M = len(self._tamps)

        grads = np.zeros(M)

        if params is None:
            Utot = self.build_Uvqc()
        else:
            Utot = self.build_Uvqc(params)

        if not self._is_multi_state:
            qc_psi = self.get_initial_computer()
            qc_psi.apply_circuit(Utot)
            qc_sig = qf.Computer(qc_psi)
            qc_sig.apply_operator(self._qb_ham)
            qc_temp = qf.Computer(qc_psi)

            mu = M - 1
            # find <sig_N | K_N | psi_N>
            Kmu_prev = self._pool_obj[self._tops[mu]][1].jw_transform(
                self._qubit_excitations
            )

            Kmu_prev.mult_coeffs(self._pool_obj[self._tops[mu]][0])

            qc_temp.apply_operator(Kmu_prev)
            grads[mu] = 2.0 * np.real(
                np.vdot(qc_sig.get_coeff_vec(), qc_temp.get_coeff_vec())
            )

            for mu in reversed(range(M - 1)):
                qc_temp = qf.Computer(qc_psi)
                # mu => N-1 => M-2
                # mu+1 => N => M-1
                # Kmu => KN-1
                # Kmu_prev => KN

                if params is None:
                    tamp = self._tamps[mu + 1]
                else:
                    tamp = params[mu + 1]

                Kmu = self._pool_obj[self._tops[mu]][1].jw_transform(
                    self._qubit_excitations
                )
                Kmu.mult_coeffs(self._pool_obj[self._tops[mu]][0])

                if self._compact_excitations:
                    if self._pool_type == "sa_SD":
                        sa_sq_op = self._pool_obj[self._tops[mu + 1]][1].terms()
                        half_length = len(sa_sq_op) // 2
                        Umu = qf.Circuit()
                        for coeff, cr, ann in sa_sq_op[:half_length]:
                            # The minus sign is dictated by the recursive algorithm used to compute the analytic gradient
                            # (see original ADAPT-VQE paper)
                            # In this particular case, the minus sign is already incorporated
                            Umu.add(
                                compact_excitation_circuit(
                                    tamp * coeff, ann, cr, self._qubit_excitations
                                )
                            )
                    else:
                        Umu = qf.Circuit()
                        # The minus sign is dictated by the recursive algorithm used to compute the analytic gradient
                        # (see original ADAPT-VQE paper)
                        Umu.add(
                            compact_excitation_circuit(
                                -tamp
                                * self._pool_obj[self._tops[mu + 1]][1].terms()[1][0],
                                self._pool_obj[self._tops[mu + 1]][1].terms()[1][1],
                                self._pool_obj[self._tops[mu + 1]][1].terms()[1][2],
                                self._qubit_excitations,
                            )
                        )
                else:
                    if self._pool_type == "sa_SD":
                        sa_sq_op = self._pool_obj[self._tops[mu + 1]][1].terms()
                        half_length = len(sa_sq_op) // 2
                        Umu = qf.Circuit()
                        for coeff, cr, ann in sa_sq_op[:half_length]:
                            sq_op = qf.SQOperator()
                            sq_op.add_term(coeff, cr, ann)
                            sq_op.add_term(-coeff, ann, cr)
                            q_op = sq_op.jw_transform(self._qubit_excitations)
                            U, p = trotterize(
                                q_op, factor=-tamp, trotter_number=self._trotter_number
                            )
                            if p != 1.0 + 0.0j:
                                raise ValueError(
                                    "Encountered phase change, phase not equal to (1.0 + 0.0i)"
                                )
                            Umu.add(U)
                    else:
                        # The minus sign is dictated by the recursive algorithm used to compute the analytic gradient
                        # (see original ADAPT-VQE paper)
                        Umu, pmu = trotterize(
                            Kmu_prev, factor=-tamp, trotter_number=self._trotter_number
                        )

                        if pmu != 1.0 + 0.0j:
                            raise ValueError(
                                "Encountered phase change, phase not equal to (1.0 + 0.0i)"
                            )

                qc_sig.apply_circuit(Umu)
                qc_psi.apply_circuit(Umu)
                qc_temp = qf.Computer(qc_psi)

                qc_temp.apply_operator(Kmu)
                grads[mu] = 2.0 * np.real(
                    np.vdot(qc_sig.get_coeff_vec(), qc_temp.get_coeff_vec())
                )

                # reset Kmu |psi_i> -> |psi_i>
                Kmu_prev = Kmu

        else:
            # TODO add sa-SD
            try:
                assert self._pool_type != "sa_SD"
            except:
                raise ValueError("sa SD not implemented for multireference ADAPT")
            # Build all Kmus and Umus in advance.
            Kmus = []
            Umus = []
            for mu in range(len(self._tops)):
                Kmu = self._pool_obj[self._tops[mu]][1].jw_transform(
                    self._qubit_excitations
                )
                Kmus.append(Kmu)
                if params is None:
                    tamp = self._tamps[mu]
                else:
                    tamp = params[mu]
                if self._compact_excitations:
                    Umu = qf.Circuit()
                    Umu.add(
                        compact_excitation_circuit(
                            -tamp * self._pool_obj[self._tops[mu]][1].terms()[1][0],
                            self._pool_obj[self._tops[mu]][1].terms()[1][1],
                            self._pool_obj[self._tops[mu]][1].terms()[1][2],
                            self._qubit_excitations,
                        )
                    )
                else:
                    Umu, pmu = trotterize(
                        Kmu, factor=-tamp, trotter_number=self._trotter_number
                    )

                    if pmu != 1.0 + 0.0j:
                        raise ValueError(
                            "Encountered phase change, phase not equal to (1.0 + 0.0i)"
                        )
                Umus.append(Umu)

            grads = np.zeros(len(self._tops))
            # print('----')
            for r in range(len(self._weights)):
                qc_psi = self.get_initial_computer()[r]
                qc_psi.apply_circuit(Utot[r])
                qc_sig = qf.Computer(qc_psi)
                qc_sig.apply_operator(self._qb_ham)
                qc_temp = qf.Computer(qc_psi)
                qc_temp.apply_operator(Kmus[M - 1])
                grads[M - 1] += (
                    2
                    * self._weights[r]
                    * np.vdot(qc_sig.get_coeff_vec(), qc_temp.get_coeff_vec()).real
                )

                for mu in reversed(range(M - 1)):
                    qc_psi.apply_circuit(Umus[mu])
                    qc_sig.apply_circuit(Umus[mu])
                    qc_temp = qf.Computer(qc_psi)
                    qc_temp.apply_operator(Kmus[mu])
                    grads[mu] += (
                        2
                        * self._weights[r]
                        * np.vdot(qc_sig.get_coeff_vec(), qc_temp.get_coeff_vec()).real
                    )

        np.testing.assert_allclose(np.imag(grads), np.zeros_like(grads), atol=1e-12)
        # print(f"Gradient: {grads}")
        # print(f"Energy: {self.measure_energy(Utot)}")

        return grads

    def measure_gradient3(self):
        """Calculates 2 Re <Psi|H K_mu |Psi> for all K_mu in self._pool_obj.
        For antihermitian K_mu, this is equal to <Psi|[H, K_mu]|Psi>.
        In ADAPT-VQE, this is the 'residual gradient' used to determine
        whether to append exp(t_mu K_mu) to the iterative ansatz.

        In the case where _is_multi_state, this will give the weighted average of
        these gradients for each reference.
        """

        if not self._fast:
            raise ValueError("self._fast must be True for gradient measurement.")

        if not self._is_multi_state:
            Utot = self.build_Uvqc()
            qc_psi = self.get_initial_computer()
            qc_psi.apply_circuit(Utot)

            qc_sig = qforte.Computer(qc_psi)
            qc_sig.apply_operator(self._qb_ham)

            grads = np.zeros(len(self._pool_obj))

            for mu, (coeff, operator) in enumerate(self._pool_obj):
                qc_temp = qf.Computer(qc_psi)
                Kmu = operator.jw_transform(self._qubit_excitations)
                Kmu.mult_coeffs(coeff)
                qc_temp.apply_operator(Kmu)
                grads[mu] = 2.0 * np.real(
                    np.vdot(qc_sig.get_coeff_vec(), qc_temp.get_coeff_vec())
                )
        else:
            Kmus = []
            for mu, (coeff, operator) in enumerate(self._pool_obj):
                Kmu = operator.jw_transform(self._qubit_excitations)
                Kmu.mult_coeffs(coeff)
                Kmus.append(Kmu)

            U_ansatz = self.ansatz_circuit()
            grads = np.zeros(len(self._pool_obj))

            for r in range(len(self._ref)):
                qc_psi = self.get_initial_computer()[r]
                qc_psi.apply_circuit(self._Uprep[r])
                qc_psi.apply_circuit(U_ansatz)
                psi_i = qc_psi.get_coeff_vec()

                qc_sig = qforte.Computer(self._nqb)
                qc_sig.set_coeff_vec(psi_i)
                qc_sig.apply_operator(self._qb_ham)

                for mu, (coeff, operator) in enumerate(self._pool_obj):
                    Kmu = Kmus[mu]
                    qc_psi.apply_operator(Kmu)
                    grads[mu] += (
                        self._weights[r]
                        * 2.0
                        * np.real(
                            np.vdot(qc_sig.get_coeff_vec(), qc_psi.get_coeff_vec())
                        )
                    )
                    qc_psi.set_coeff_vec(psi_i)

        np.testing.assert_allclose(np.imag(grads), np.zeros_like(grads), atol=1e-7)

        return grads

    def gradient_ary_feval(self, params):
        grads = self.measure_gradient(params)

        if self._noise_factor > 1e-14:
            grads = [
                np.random.normal(np.real(grad_m), self._noise_factor)
                for grad_m in grads
            ]

        if not self._is_multi_state:
            factor = 1
        else:
            factor = len(self._ref)
        self._curr_grad_norm = np.linalg.norm(grads)
        self._res_vec_evals += factor
        self._res_m_evals += factor * len(self._tamps)

        return np.asarray(grads)

    def report_iteration(self, x):
        self._k_counter += 1

        if self._k_counter == 1:
            header = "\n    k iteration         Energy               dE"
            if self._use_analytic_grad:
                header += "           Ngvec ev      Ngm ev*         ||g||"
            header += "\n------------------------------------------------------"
            if self._use_analytic_grad:
                header += "--------------------------------------------"
            print(header)
            if self._print_summary_file:
                header.replace("\n ", "\n#  ").replace("\n-", "\n#--")
                with open("summary.dat", "w+", buffering=1) as f:
                    f.write(header)

        # else:
        dE = self._curr_energy - self._prev_energy
        update = f"     {self._k_counter:7}        {self._curr_energy:+12.10f}      {dE:+12.10f}"
        if self._use_analytic_grad:
            update += f"      {self._res_vec_evals:4}        {self._res_m_evals:6}       {self._curr_grad_norm:+12.10f}"
        print(update)

        if self._print_summary_file:
            with open("summary.dat", "a", buffering=1) as f:
                update = "\n  " + update
                f.write(header)

        self._prev_energy = self._curr_energy

    def verify_required_UCCVQE_attributes(self):
        if self._use_analytic_grad is None:
            raise NotImplementedError(
                "Concrete UCCVQE class must define self._use_analytic_grad attribute."
            )

        if self._pool_type is None:
            raise NotImplementedError(
                "Concrete UCCVQE class must define self._pool_type attribute."
            )

        if self._pool_obj is None:
            raise NotImplementedError(
                "Concrete UCCVQE class must define self._pool_obj attribute."
            )
