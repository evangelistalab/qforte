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

class UCCVQE(VQE, UCC):
    """The abstract base class inheritied by any algorithm that seeks to find
    eigenstates by variational minimization of the Energy

    .. math::
        E(\mathbf{t}) = \langle \Phi_0 | \hat{U}^\dagger(\mathbf{\mathbf{t}}) \hat{H} \hat{U}(\mathbf{\mathbf{t}}) | \Phi_0 \\rangle

    using a disentagled UCC type ansatz

    .. math::
        \hat{U}(\mathbf{t}) = \prod_\mu e^{t_\mu (\hat{\\tau}_\mu - \hat{\\tau}_\mu^\dagger)},

    were :math:`\hat{\\tau}_\mu` is a Fermionic excitation operator and
    :math:`t_\mu` is a cluster amplitude.

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

    @abstractmethod
    def get_num_ham_measurements(self):
        pass

    @abstractmethod
    def get_num_commut_measurements(self):
        pass

    def fill_commutator_pool(self):
        print('\n\n==> Building commutator pool for gradient measurement.')
        self._commutator_pool = self._pool_obj.get_qubit_op_pool()
        self._commutator_pool.join_as_commutator(self._qb_ham)
        print('==> Commutator pool construction complete.')

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
            myQC = qforte.Computer(self._nqb)
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
        if(self._computer_type == 'fock'):
            return self.measure_gradient_fock(params)
        elif(self._computer_type == 'fci'):
            return self.measure_gradient_fci(params)
        else:
            raise ValueError(f"{self._computer_type} is an unrecognized computer type.") 

    def measure_gradient_fock(self, params=None):
        """ Returns the disentangled (factorized) UCC gradient, using a
        recursive approach.

        Parameters
        ----------
        params : list of floats
            The variational parameters which characterize _Uvqc.
        """

        if not self._fast:
            raise ValueError("self._fast must be True for gradient measurement.")

        M = len(self._tamps)

        grads = np.zeros(M)

        # print(f"\n Grads before: {grads}")

        if params is None:
            Utot = self.build_Uvqc()
        else:
            Utot = self.build_Uvqc(params)

        qc_psi = qforte.Computer(self._nqb) # build | sig_N > according ADAPT-VQE analytical grad section
        qc_psi.apply_circuit(Utot)
        qc_sig = qforte.Computer(self._nqb) # build | psi_N > according ADAPT-VQE analytical grad section
        psi_i = copy.deepcopy(qc_psi.get_coeff_vec())
        qc_sig.set_coeff_vec(copy.deepcopy(psi_i)) # not sure if copy is faster or reapplication of state
        qc_sig.apply_operator(self._qb_ham)

        mu = M-1

        # find <sing_N | K_N | psi_N>
        Kmu_prev = self._pool_obj[self._tops[mu]][1].jw_transform(self._qubit_excitations)
        Kmu_prev.mult_coeffs(self._pool_obj[self._tops[mu]][0])

        qc_psi.apply_operator(Kmu_prev)
        grads[mu] = 2.0 * np.real(np.vdot(qc_sig.get_coeff_vec(), qc_psi.get_coeff_vec()))

        #reset Kmu_prev |psi_i> -> |psi_i>
        qc_psi.set_coeff_vec(copy.deepcopy(psi_i))

        for mu in reversed(range(M-1)):

            # mu => N-1 => M-2
            # mu+1 => N => M-1
            # Kmu => KN-1
            # Kmu_prev => KN

            if params is None:
                tamp = self._tamps[mu+1]
            else:
                tamp = params[mu+1]

            Kmu = self._pool_obj[self._tops[mu]][1].jw_transform(self._qubit_excitations)
            Kmu.mult_coeffs(self._pool_obj[self._tops[mu]][0])

            if self._compact_excitations:
                Umu = qf.Circuit()
                # The minus sign is dictated by the recursive algorithm used to compute the analytic gradient
                # (see original ADAPT-VQE paper)
                Umu.add(compact_excitation_circuit(-tamp * self._pool_obj[self._tops[mu + 1]][1].terms()[1][0],
                                                           self._pool_obj[self._tops[mu + 1]][1].terms()[1][1],
                                                           self._pool_obj[self._tops[mu + 1]][1].terms()[1][2],
                                                           self._qubit_excitations))
            else:
                # The minus sign is dictated by the recursive algorithm used to compute the analytic gradient
                # (see original ADAPT-VQE paper)
                Umu, pmu = trotterize(Kmu_prev, factor=-tamp, trotter_number=self._trotter_number)

                if (pmu != 1.0 + 0.0j):
                    raise ValueError("Encountered phase change, phase not equal to (1.0 + 0.0i)")

            qc_sig.apply_circuit(Umu)
            qc_psi.apply_circuit(Umu)
            psi_i = copy.deepcopy(qc_psi.get_coeff_vec())

            qc_psi.apply_operator(Kmu)
            grads[mu] = 2.0 * np.real(np.vdot(qc_sig.get_coeff_vec(), qc_psi.get_coeff_vec()))

            #reset Kmu |psi_i> -> |psi_i>
            qc_psi.set_coeff_vec(copy.deepcopy(psi_i))
            Kmu_prev = Kmu

        np.testing.assert_allclose(np.imag(grads), np.zeros_like(grads), atol=1e-7)

        # print(f"\n Grads after: {grads}")

        return grads
    
    def measure_gradient_fci(self, params=None):
        """ Returns the disentangled (factorized) UCC gradient, using a
        recursive approach.

        Parameters
        ----------
        params : list of floats
            The variational parameters which characterize _Uvqc.
        """

        if not self._fast:
            raise ValueError("self._fast must be True for gradient measurement.")
        
        if(self._pool_type == 'sa_SD'):
            raise ValueError('Must use single term particle-hole nbody operators for residual calculation')
        
        if not self._ref_from_hf:
            raise ValueError('get_residual_vector_fci_comp only compatible with hf reference at this time.')

        M = len(self._tamps)
        grads = np.zeros(M)
        vqc_ops = qforte.SQOpPool()

        if params is None:
            for tamp, top in zip(self._tamps, self._tops):
                vqc_ops.add(tamp, self._pool_obj[top][1])
        else:
            for tamp, top in zip(params, self._tops):
                vqc_ops.add(tamp, self._pool_obj[top][1])

        # build | sig_N > according ADAPT-VQE analytical grad section
        qc_psi = qforte.FCIComputer(
            self._nel, 
            self._2_spin, 
            self._norb) 
        
        qc_psi.hartree_fock()
        
        # qc_psi.apply_circuit(Utot)
        qc_psi.evolve_pool_trotter_basic(
            vqc_ops,
            antiherm=True,
            adjoint=False)

        # build | psi_N > according ADAPT-VQE analytical grad section
        qc_sig = qforte.FCIComputer(
            self._nel, 
            self._2_spin, 
            self._norb) 

        psi_i = qc_psi.get_state_deep()

        # not sure if copy is faster or reapplication of state
        qc_sig.set_state(psi_i) 

        if(self._apply_ham_as_tensor):
            qc_sig.apply_tensor_spat_012bdy(
                self._nuclear_repulsion_energy, 
                self._mo_oeis, 
                self._mo_teis, 
                self._mo_teis_einsum, 
                self._norb)
        else:   
            qc_sig.apply_sqop(self._sq_ham)

        mu = M-1

        # find <sing_N | K_N | psi_N>
        Kmu_prev = self._pool_obj[self._tops[mu]][1]

        Kmu_prev.mult_coeffs(self._pool_obj[self._tops[mu]][0])

        qc_psi.apply_sqop(Kmu_prev)
        grads[mu] = 2.0 * np.real(
            qc_sig.get_state().vector_dot(qc_psi.get_state())
            )

        #reset Kmu_prev |psi_i> -> |psi_i>
        qc_psi.set_state(psi_i)

        for mu in reversed(range(M-1)):

            # mu => N-1 => M-2
            # mu+1 => N => M-1
            # Kmu => KN-1
            # Kmu_prev => KN

            if params is None:
                tamp = self._tamps[mu+1]
            else:
                tamp = params[mu+1]

            # Kmu = self._pool_obj[self._tops[mu]][1].jw_transform(self._qubit_excitations)
            Kmu = self._pool_obj[self._tops[mu]][1]

            Kmu.mult_coeffs(self._pool_obj[self._tops[mu]][0])

            # The minus sign is dictated by the recursive algorithm used to compute the analytic gradient
            # (see original ADAPT-VQE paper)
            qc_psi.apply_sqop_evolution(
                -1.0*tamp,
                Kmu_prev,
                antiherm=True,
                adjoint=False)
            
            qc_sig.apply_sqop_evolution(
                -1.0*tamp,
                Kmu_prev,
                antiherm=True,
                adjoint=False)

            psi_i = qc_psi.get_state_deep()

            qc_psi.apply_sqop(Kmu)
            grads[mu] = 2.0 * np.real(
                qc_sig.get_state().vector_dot(qc_psi.get_state())
                )

            #reset Kmu |psi_i> -> |psi_i>
            qc_psi.set_state(psi_i)
            Kmu_prev = Kmu

        np.testing.assert_allclose(np.imag(grads), np.zeros_like(grads), atol=1e-7)
        return grads

    def measure_gradient3(self):
        """ Calculates 2 Re <Psi|H K_mu |Psi> for all K_mu in self._pool_obj.
        For antihermitian K_mu, this is equal to <Psi|[H, K_mu]|Psi>.
        In ADAPT-VQE, this is the 'residual gradient' used to determine
        whether to append exp(t_mu K_mu) to the iterative ansatz.
        """

        if not self._fast:
            raise ValueError("self._fast must be True for gradient measurement.")

        Utot = self.build_Uvqc()
        qc_psi = qforte.Computer(self._nqb)
        qc_psi.apply_circuit(Utot)
        psi_i = copy.deepcopy(qc_psi.get_coeff_vec())

        qc_sig = qforte.Computer(self._nqb)
        # TODO: Check if it's faster to recompute psi_i or copy it.
        qc_sig.set_coeff_vec(copy.deepcopy(psi_i))
        qc_sig.apply_operator(self._qb_ham)

        grads = np.zeros(len(self._pool_obj))

        for mu, (coeff, operator) in enumerate(self._pool_obj):
            Kmu = operator.jw_transform(self._qubit_excitations)
            Kmu.mult_coeffs(coeff)
            qc_psi.apply_operator(Kmu)
            grads[mu] = 2.0 * np.real(np.vdot(qc_sig.get_coeff_vec(), qc_psi.get_coeff_vec()))
            qc_psi.set_coeff_vec(copy.deepcopy(psi_i))

        np.testing.assert_allclose(np.imag(grads), np.zeros_like(grads), atol=1e-7)

        return grads

    def gradient_ary_feval(self, params):
        if(self._computer_type == 'fock'):
            return self.gradient_ary_feval_fock(params)
        elif(self._computer_type == 'fci'):
            return self.gradient_ary_feval_fci(params)
        else:
            raise ValueError(f"{self._computer_type} is an unrecognized computer type.") 

    def gradient_ary_feval_fock(self, params):
        grads = self.measure_gradient(params)

        if(self._noise_factor > 1e-14):
            grads = [np.random.normal(np.real(grad_m), self._noise_factor) for grad_m in grads]

        self._curr_grad_norm = np.linalg.norm(grads)
        self._res_vec_evals += 1
        self._res_m_evals += len(self._tamps)

        return np.asarray(grads)
    
    def gradient_ary_feval_fci(self, params):
        grads = self.measure_gradient(params)

        if(self._noise_factor > 1e-14):
            grads = [np.random.normal(np.real(grad_m), self._noise_factor) for grad_m in grads]

        self._curr_grad_norm = np.linalg.norm(grads)
        self._res_vec_evals += 1
        self._res_m_evals += len(self._tamps)

        return np.asarray(grads)

    def report_iteration(self, x):

        self._k_counter += 1

        if(self._k_counter == 1):
            print('\n    k iteration         Energy               dE           Ngvec ev      Ngm ev*         ||g||')
            print('--------------------------------------------------------------------------------------------------')
            if (self._print_summary_file):
                f = open("summary.dat", "w+", buffering=1)
                f.write('\n#    k iteration         Energy               dE           Ngvec ev      Ngm ev*         ||g||')
                f.write('\n#--------------------------------------------------------------------------------------------------')
                f.close()

        # else:
        dE = self._curr_energy - self._prev_energy
        print(f'     {self._k_counter:7}        {self._curr_energy:+12.10f}      {dE:+12.10f}      {self._res_vec_evals:4}        {self._res_m_evals:6}       {self._curr_grad_norm:+12.10f}')

        if (self._print_summary_file):
            f = open("summary.dat", "a", buffering=1)
            f.write(f'\n       {self._k_counter:7}        {self._curr_energy:+12.12f}      {dE:+12.12f}      {self._res_vec_evals:4}        {self._res_m_evals:6}       {self._curr_grad_norm:+12.12f}')
            f.close()

        self._prev_energy = self._curr_energy

    def verify_required_UCCVQE_attributes(self):
        if self._use_analytic_grad is None:
            raise NotImplementedError('Concrete UCCVQE class must define self._use_analytic_grad attribute.')

        if self._pool_type is None:
            raise NotImplementedError('Concrete UCCVQE class must define self._pool_type attribute.')

        if self._pool_obj is None:
            raise NotImplementedError('Concrete UCCVQE class must define self._pool_obj attribute.')
