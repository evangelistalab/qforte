"""
QITE classes
=================================================
Classes for using a quantum computer to carry
out the quantum imaginary time evolution algorithm.
"""
import qforte as qf
from qforte.abc.algorithm import Algorithm
from qforte.utils.transforms import (get_jw_organizer,
                                    organizer_to_circuit)

from qforte.utils.state_prep import *
from qforte.utils.trotterization import trotterize
from qforte.helper.printing import *
import copy
import numpy as np
from scipy.linalg import lstsq

from qforte.maths.eigsolve import canonical_geig_solve

### Throughout this file, we'll refer to DOI 10.1038/s41567-019-0704-4 as Motta.

class QITE(Algorithm):
    """This class implements the quantum imaginary time evolution (QITE)
    algorithm in a fashion amenable to non k-local hamiltonains, which is
    the focus of the origional algorithm (see DOI 10.1038/s41567-019-0704-4).

    In QITE one attepmts to approximate the action of the imaginary time evolution
    operator on a state :math:`| \Phi \\rangle` with a parameterized unitary
    operation:

    .. math::
        c(\\Delta \\beta)^{-1/2} e^{-\\Delta \\beta \hat{H}} | \Phi \\rangle \\approx e^{-i \\Delta \\beta \hat{A}(\\vec{\\theta})} | \Phi \\rangle,

    where :math:`\\Delta \\beta` is a small time step and
    :math:`c(\\Delta \\beta)^{-1/2}` is a normalization coefficient approximated
    by :math:`1-2\\Delta \\beta \\langle \Phi | \hat{H} | \Phi \\rangle`.

    The parameterized anti-hermetian operator :math:`\hat{A}(\\vec{\\theta})`
    is given by the linear combination of :math:`N_\mu` operators

    .. math::
        \hat{A}(\\vec{\\theta}) = \sum_\mu^{N_\mu} \\theta_\mu \hat{P}_\mu,

    where :math:`\hat{P}_\mu` is a product of Pauli operators. In practice the
    operators that enter in to the sum are a subset of an operator pool specified
    by the user.

    To determine the parameters :math:`\\theta_\mu` one seeks to satisfy the
    condition:

    .. math::
        c(\\beta)^{-1/2} \\langle \Phi |  \sum_{\mu} \\theta_\mu \hat{P}_\mu^\dagger \hat{H} | \Phi \\rangle
        \\approx -i  \\langle \Phi | \sum_{\mu} \\theta_\mu \\theta_\\nu \hat{P}_\mu^\dagger  \hat{P}_\\nu | \Phi \\rangle

    which corresponding to solving the linear systems

    .. math::
        \mathbf{S} \\vec{\\theta} = \\vec{b}

    where the elements

    .. math::
        S_{\mu \\nu} = \\langle \Phi | \hat{P}_\mu^\dagger \hat{P}_\\nu | \Phi \\rangle,

    .. math::
        b_\mu = \\frac{-i}{\sqrt{c(\Delta \\beta)}} \\langle \Phi | \hat{P}_\mu^\dagger \hat{H} | \Phi \\rangle

    can be measured on a quantum device.

    Note that the QITE procedure is iterative and is repated for a specified
    number of time steps to reach a target total evolution time.

    Attributes
    ----------

    _b_thresh : float
        The minimum threshold absolute vale for the elements of :math:`b_\mu` to be included
        in the solving of the linear system. Operators :math:`\hat{P}_\mu`
        corresponding to elements of :math:`|b_\mu|` < _b_thresh will not enter
        into the operator :math:`\hat{A}`.

    _x_thresh : float
        Operators :math:`\hat{P}_\mu` corresponding to elements of :math:`|\\theta_\mu|`
        < _b_thresh will not enter into the operator :math:`\hat{A}`.

    _beta : float
        The target total evolution time.

    _db : float
        The imaginary time step to use.

    _do_lanczos : bool
        Whether or not to additionaly compute the QLanczos QSD matrices and
        solve the corresponding generailzed eigenvalue problem.

    _Ekb : list of float
        The list of after each additional time step.

    _expansion_type: {'complete_qubit', 'cqoy', 'SD', 'GSD', 'SDT', SDTQ', 'SDTQP', 'SDTQPH'}
        The family of operators that each evolution operator :math:`\hat{A}` will be built of.

    _lanczos_gap : int
        The number of time steps between generation of Lanczos basis vectors.

    _nbeta: int
        How many QITE steps should be taken? (not directly specified by user).

    _NI : int
        The number of operators in _sig.

    _sig : QubitOpPool
        The basis of operators allowed in a unitary evolution step.

    _sparseSb : bool
        Use sparse tensors to solve the linear system?

    _Uqite : Circuit
        The circuit that prepares the QITE state at the current iteration.


    """
    def run(self,
            beta=1.0,
            db=0.2,
            use_exact_evolution=False,
            expansion_type='SD',
            sparseSb=True,
            low_memorySb=False,
            second_order=False,
            b_thresh=1.0e-6,
            x_thresh=1.0e-10,
            do_lanczos=False,
            lanczos_gap=2,
            realistic_lanczos=True,
            fname=None):

        self._beta = beta
        self._db = db
        self._use_exact_evolution = use_exact_evolution
        self._nbeta = int(beta/db)+1
        self._expansion_type = expansion_type
        self._sparseSb = sparseSb
        self._low_memorySb = low_memorySb
        self._second_order = second_order
        self._total_phase = 1.0 + 0.0j
        self._Uqite = qf.Circuit()
        self._b_thresh = b_thresh
        self._x_thresh = x_thresh

        self._n_classical_params = 0
        self._n_cnot = self._Uprep.get_num_cnots()
        self._n_pauli_trm_measures = 0

        self._do_lanczos = do_lanczos
        self._lanczos_gap = lanczos_gap
        self._realistic_lanczos = realistic_lanczos
        self._fname = fname

        if(self._fname is None):
            if(self._use_exact_evolution):
                self._fname = f'beta_{self._beta}_db_{self._db}_EXACT_EVOLUTION'
            else:
                self._fname = f'beta_{self._beta}_db_{self._db}_{self._computer_type}_{self._expansion_type}_second_order_{self._second_order}'

        self._sz = 0

        if(self._computer_type=='fci'):
            qc_ref = qf.FCIComputer(self._nel, self._sz, self._norb)
            qc_ref.hartree_fock()

            if(self._apply_ham_as_tensor):

                self._Ekb = [np.real(qc_ref.get_exp_val_tensor(
                        self._nuclear_repulsion_energy, 
                        self._mo_oeis, 
                        self._mo_teis, 
                        self._mo_teis_einsum, 
                        self._norb))]

            else:
                self._Ekb = [np.real(qc_ref.get_exp_val(self._sq_ham))]
            
        if(self._computer_type=='fock'):
            qc_ref = qf.Computer(self._nqb)
            qc_ref.apply_circuit(self._Uprep)
            self._Ekb = [np.real(qc_ref.direct_op_exp_val(self._qb_ham))]

        # Print options banner (should done for all algorithms).
        self.print_options_banner()

        # Build expansion pool.
        if(not self._use_exact_evolution):
            self.build_expansion_pool()

        # Do the imaginary time evolution.
        timer = qf.local_timer()
        timer.reset()

        self.evolve()

        timer.record('Total evolution time')
        print(timer)

        # Print summary banner (should done for all algorithms).
        self.print_summary_banner()

        # verify that required attributes were defined
        # (should done for all algorithms).
        self.verify_run()

    def run_realistic(self):
        raise NotImplementedError('run_realistic() is not yet implemented for QITE.')

    def verify_run(self):
        self.verify_required_attributes()

    def print_options_banner(self):
        print('\n-----------------------------------------------------')
        print('     Quantum Imaginary Time Evolution Algorithm   ')
        print('-----------------------------------------------------')

        print('\n\n                 ==> QITE options <==')
        print('-----------------------------------------------------------')
        # General algorithm options.
        print('Computer Type:                           ',  self._computer_type)
        print('Trial reference state:                   ',  ref_string(self._ref, self._nqb))
        print('Number of Hamiltonian Pauli terms:       ',  self._Nl)
        print('Trial state preparation method:          ',  self._state_prep_type)
        print('Trotter order (rho):                     ',  self._trotter_order)
        print('Trotter number (m):                      ',  self._trotter_number)
        print('Use fast version of algorithm:           ',  str(self._fast))
        if(not self._fast):
            print('Measurement variance thresh:             ',  0.01)

        # Specific QITE options.
        print('Total imaginary evolution time (beta):   ',  self._beta)
        print('Imaginary time step (db):                ',  self._db)
        print('Use exact evolution:                     ',  self._use_exact_evolution)
        print('Expansion type:                          ',  self._expansion_type)
        print('x value threshold:                       ',  self._x_thresh)
        print('Use sparse tensors to solve Sx = b:      ',  str(self._sparseSb))
        if(self._sparseSb):
            print('b value threshold:                       ',  str(self._b_thresh))
        print('\n')
        print('Use low memory mode:                     ',  self._low_memorySb)
        print('Use 2nd order derivation of QITE:        ',  self._second_order)
        print('Do Quantum Lanczos                       ',  str(self._do_lanczos))
        if(self._do_lanczos):
            print('Lanczos gap size                         ',  self._lanczos_gap)

    def print_summary_banner(self):
        print('\n\n                        ==> QITE summary <==')
        print('-----------------------------------------------------------')
        print('Final QITE Energy:                        ', round(self._Egs, 10))
        if(not self._use_exact_evolution):
            print('Number of operators in pool:              ', self._NI)
            print('Number of classical parameters used:      ', self._n_classical_params)
            print('Estimated classical memory usage (GB):    ', f'{self._total_memory * 10**-9:e}')
            print('Number of CNOT gates in deepest circuit:  ', self._n_cnot)
            print('Number of Pauli term measurements:        ', self._n_pauli_trm_measures)

    def build_expansion_pool(self):
        print('\n==> Building expansion pool <==')

        if(self._computer_type=='fci'):
            self._sig = qf.SQOpPool() # changed this from QubitOpPool
            self._sig.set_orb_spaces(self._ref)

            if(self._expansion_type in {'SD', 'GSD', 'SDT', 'SDTQ', 'SDTQP', 'SDTQPH'}):
                self._sig.fill_pool(self._expansion_type) # This automatically filters non-particle conserving terms

            else:
                raise ValueError('Invalid expansion type specified.')

            self._NI = len(self._sig.terms())

        if(self._computer_type=='fock'):
            self._sig = qf.QubitOpPool()

            if(self._expansion_type == 'complete_qubit'):
                if (self._nqb > 6):
                    raise ValueError('Using complete qubits expansion will result in a very large number of terms!')
                self._sig.fill_pool("complete_qubit", self._ref)

            elif(self._expansion_type == 'cqoy'):
                self._sig.fill_pool("cqoy", self._ref)

            elif(self._expansion_type in {'SD', 'GSD', 'SDT', 'SDTQ', 'SDTQP', 'SDTQPH'}):
                P = qf.SQOpPool()
                P.set_orb_spaces(self._ref) 
                P.fill_pool(self._expansion_type)
                sig_temp = P.get_qubit_operator("commuting_grp_lex", False)

                # Filter the generated operators, so that only those with an odd number of Y gates are allowed.
                # See section "Real Hamiltonians and states" in the SI of Motta for theoretical justification.
                # Briefly, this method solves Ax=b, but all b elements with an odd number of Y gates are imaginary and
                # thus vanish. This method will not be correct for non-real Hamiltonians or states.
                for _, rho in sig_temp.terms():
                    nygates = 0
                    temp_rho = qf.Circuit()
                    for gate in rho.gates():
                        temp_rho.add(qf.gate(gate.gate_id(), gate.target(), gate.control()))
                        if (gate.gate_id() == "Y"):
                            nygates += 1

                    if (nygates % 2 == 1):
                        rho_op = qf.QubitOperator()
                        rho_op.add(1.0, temp_rho)
                        self._sig.add(1.0, rho_op)

            else:
                raise ValueError('Invalid expansion type specified.')

            self._NI = len(self._sig.terms())


    def build_S_b_FCI(self):
        """Construct the matrix S (eq. 5a) and vector b (eq. 5b) of Motta, with h[m] the full Hamiltonian, utilizing FCIComputer class.
        """
        Idim = self._NI

        self._n_pauli_trm_measures += int(self._NI*(self._NI+1)*0.5)
        self._n_pauli_trm_measures += self._Nl * self._NI

        # Initialize linear system
        S = np.zeros((Idim, Idim), dtype=complex)
        b = np.zeros(Idim, dtype=complex)

        if(self._second_order):
            prefactor = -2.0
            S_factor = 2.0

        else:
            denom = np.sqrt(1.0 - 2.0*self._db*self._Ekb[-1])
            prefactor = -1.0 / denom
            S_factor = 1.0

        Ipsi_qc = qf.FCIComputer(self._nel, self._sz, self._norb)
        Hpsi_qc = qf.FCIComputer(self._nel, self._sz, self._norb)
        Hpsi_qc.set_state(self._qc.get_state_deep())
 
        if(self._apply_ham_as_tensor):
            Hpsi_qc.apply_tensor_spat_012bdy(
                    self._nuclear_repulsion_energy, 
                    self._mo_oeis, 
                    self._mo_teis, 
                    self._mo_teis_einsum, 
                    self._norb)
        else:
            Hpsi_qc.apply_sqop(self._sq_ham)

        if(self._low_memorySb):
            for i in range(Idim):
                S[i][i] = 1.0 # With Pauli strings, this is always the inner product

                # initialize state and apply pool term
                Ipsi_qc.set_state(self._qc.get_state_deep())
                Ipsi_qc.apply_sqop(self._sig.terms()[i][1])
                Ipsi_mu = Ipsi_qc.get_state_deep()

                # build b (second order variation)
                if(self._second_order):
                    exp_val = Hpsi_qc.get_state_deep().vector_dot(Ipsi_mu)
                    b[i] = prefactor * exp_val

                # build b (original)
                else:
                    exp_val = Ipsi_mu.vector_dot(Hpsi_qc.get_state_deep())
                    b[i] = prefactor * exp_val

                # populate lower triangle of S and copy conjugate to upper triangle
                for j in range(i):
                    Ipsi_qc.set_state(self._qc.get_state_deep())
                    Ipsi_qc.apply_sqop(self._sig.terms()[j][1])

                    S[i][j] = Ipsi_mu.vector_dot(Ipsi_qc.get_state_deep())
                    S[j][i] = S[i][j].conj()

            return S_factor * np.real(S), np.real(b)

        else:
            rho_psi = []
            for i in range(Idim):
                S[i][i] = 1.0 # With Pauli strings, this is always the inner product
                
                # initialize state and apply pool term
                Ipsi_qc.set_state(self._qc.get_state_deep())
                Ipsi_qc.apply_sqop(self._sig.terms()[i][1])
                rho_psi.append(Ipsi_qc.get_state_deep())

                # build b (second order variation)
                if(self._second_order):
                    exp_val = Hpsi_qc.get_state_deep().vector_dot(rho_psi[i])
                    b[i] = prefactor * exp_val

                # build b (original)
                else:
                    exp_val = rho_psi[i].vector_dot(Hpsi_qc.get_state_deep())
                    b[i] = prefactor * exp_val

                # populate lower triangle of S and copy conjugate to upper triangle
                for j in range(i):
                    S[i][j] = rho_psi[i].vector_dot(rho_psi[j])
                    S[j][i] = S[i][j].conj()

            return S_factor * np.real(S), np.real(b)


    def build_S(self):
        """Construct the matrix S (eq. 5a) of Motta.
        """
        Idim = self._NI

        S = np.zeros((Idim, Idim), dtype=complex)

        Ipsi_qc = qf.Computer(self._nqb)
        Ipsi_qc.set_coeff_vec(copy.deepcopy(self._qc.get_coeff_vec()))
        # CI[I][J] = (σ_I Ψ)_J
        self._n_pauli_trm_measures += int(self._NI*(self._NI+1)*0.5)
        CI = np.zeros(shape=(Idim, int(2**self._nqb)), dtype=complex)

        for i in range(Idim):
            S[i][i] = 1.0 # With Pauli strings, this is always the inner product
            Ipsi_qc.apply_operator(self._sig.terms()[i][1])
            CI[i,:] = copy.deepcopy(Ipsi_qc.get_coeff_vec())
            for j in range(i):
                S[i][j] = S[j][i] = np.vdot(CI[i,:], CI[j,:])
            Ipsi_qc.set_coeff_vec(copy.deepcopy(self._qc.get_coeff_vec()))

        return np.real(S)


    def build_sparse_S_b(self, b):
        b_sparse = []
        idx_sparse = []
        for I, bI in enumerate(b):
            if(np.abs(bI) > self._b_thresh):
                idx_sparse.append(I)
                b_sparse.append(bI)
        Idim = len(idx_sparse)
        self._n_pauli_trm_measures += int(Idim*(Idim+1)*0.5)

        S = np.zeros((len(b_sparse),len(b_sparse)), dtype=complex)

        Ipsi_qc = qf.Computer(self._nqb)
        Ipsi_qc.set_coeff_vec(copy.deepcopy(self._qc.get_coeff_vec()))
        CI = np.zeros(shape=(Idim, int(2**self._nqb)), dtype=complex)

        for i in range(Idim):
            S[i][i] = 1.0 # With Pauli strings, this is always the inner product
            Ii = idx_sparse[i]
            Ipsi_qc.apply_operator(self._sig.terms()[Ii][1])
            CI[i,:] = copy.deepcopy(Ipsi_qc.get_coeff_vec())
            for j in range(i):
                S[i][j] = S[j][i] = np.vdot(CI[i,:], CI[j,:])
            Ipsi_qc.set_coeff_vec(copy.deepcopy(self._qc.get_coeff_vec()))

        return idx_sparse, np.real(S), np.real(b_sparse)

    def build_b(self):
        """Construct the vector b (eq. 5b) of Motta, with h[m] the full Hamiltonian.
        """

        b  = np.zeros(self._NI, dtype=complex)

        denom = np.sqrt(1 - 2*self._db*self._Ekb[-1])
        prefactor = -1.0j / denom

        self._n_pauli_trm_measures += self._Nl * self._NI

        Hpsi_qc = qf.Computer(self._nqb)
        Hpsi_qc.set_coeff_vec(copy.deepcopy(self._qc.get_coeff_vec()))
        Hpsi_qc.apply_operator(self._qb_ham)
        C_Hpsi_qc = copy.deepcopy(Hpsi_qc.get_coeff_vec())

        for I, (op_coefficient, operator) in enumerate(self._sig.terms()):
            Hpsi_qc.apply_operator(operator)
            exp_val = np.vdot(self._qc.get_coeff_vec(), Hpsi_qc.get_coeff_vec())
            b[I] = prefactor * op_coefficient * exp_val
            Hpsi_qc.set_coeff_vec(copy.deepcopy(C_Hpsi_qc))

        return np.real(b)


    def do_qite_step(self):

        if(self._computer_type=='fci'):
            if(self._sparseSb):
                print(f"Warning, build sparseSb method isn't supported for FCI computer, setting option to false")
                self._sparseSb = False

            S, btot = self.build_S_b_FCI()

        if(self._computer_type=='fock'):
            if(self._low_memorySb):
                print(f"Warning, build low memory Sb method isn't supported for Fock computer, setting option to false")
                self._low_memorySb = False

            btot = self.build_b()
            A = qf.QubitOperator()

            if(self._sparseSb):
                sp_idxs, S, btot = self.build_sparse_S_b(btot)
            else:
                S = self.build_S()

        x = lstsq(S, btot)[0]
        x = np.real(x)
        x_list = x.tolist()
        # this is only for UCC!
        x_list_fci = [x*self._db for x in x_list]

        if(self._computer_type=='fock'):
            if(self._sparseSb):
                for I, spI in enumerate(sp_idxs):
                    if np.abs(x[I]) > self._x_thresh:
                        A.add(-1.0j * self._db * x[I], self._sig.terms()[spI][1].terms()[0][1])
                        self._n_classical_params += 1

            else:
                for I, SigI in enumerate(self._sig.terms()):
                    if np.abs(x[I]) > self._x_thresh:
                        A.add(-1.0j * self._db * x[I], SigI[1].terms()[0][1])
                        self._n_classical_params += 1

        if(self._verbose):
            print('\nbtot:\n ', btot)
            print('\n S:  \n')
            matprint(S)
            print('\n x:  \n')
            print(x)

        # added fock computer conditional
        if(self._computer_type=='fock'):
            eiA_kb, phase1 = trotterize(A, trotter_number=self._trotter_number)
            self._total_phase *= phase1
            self._Uqite.add(eiA_kb)
            self._qc.apply_circuit(eiA_kb)
            self._Ekb.append(np.real(self._qc.direct_op_exp_val(self._qb_ham)))

            self._n_cnot += eiA_kb.get_num_cnots()

        if(self._computer_type=='fci'):
            self._sig.set_coeffs(x_list_fci)
            self._qc.evolve_pool_trotter_basic(self._sig, True, False)

            if(self._apply_ham_as_tensor):
                self._Ekb.append(np.real(self._qc.get_exp_val_tensor(
                        self._nuclear_repulsion_energy, 
                        self._mo_oeis, 
                        self._mo_teis, 
                        self._mo_teis_einsum, 
                        self._norb)))
            else:
                self._Ekb.append(np.real(self._qc.get_exp_val(self._sq_ham)))


        if(self._verbose):
            qf.smart_print(self._qc)

    def evolve(self):
        """Perform QITE for a time step :math:`\\Delta \\beta`.
        """
    
        if(self._computer_type=='fock'):
            self._Uqite.add(self._Uprep)
            self._qc = qf.Computer(self._nqb)
            self._qc.apply_circuit(self._Uqite)

        if(self._computer_type=='fci'):
            self._qc = qf.FCIComputer(self._nel, self._sz, self._norb)
            self._qc.hartree_fock()

            if(not self._use_exact_evolution):
                qc_size = self._qc.get_state().size()
                if(self._low_memorySb):
                    self._total_memory = 5.0 * 16.0 * qc_size
                else:
                    self._total_memory = self._NI * 16.0 * qc_size

                if(self._total_memory > 8.0e9 and not self._low_memorySb):
                    print('\n')
                    print('WARNING: ESTIMATED MEMORY USAGE EXCEEDS 8GB, SWITCHING TO LOW MEMORY MODE')
                    print('\n')
                    self._low_memorySb = True
                    self._total_memory = 5.0 * 16.0 * qc_size # 5 corresponds to total # of tensors at any given time in memory


        if(self._do_lanczos and not self._use_exact_evolution):
            #initialize constant list to build H and S matricies
            if(self._realistic_lanczos):
                self._c_list = []
                self._c_list.append(1.0) # will always be 1.0 for 0th iteration (E_l = E_0)

            else:
                self._lanczos_vecs = []
                self._Hlanczos_vecs = []
                self._lanczos_vecs.append(self._qc.get_state_deep())

                qcSig_temp = qf.FCIComputer(self._nel, self._sz, self._norb)
                qcSig_temp.set_state(self._qc.get_state_deep())

                if(self._apply_ham_as_tensor):
                    qcSig_temp.apply_tensor_spat_012bdy(
                            self._nuclear_repulsion_energy, 
                            self._mo_oeis, 
                            self._mo_teis, 
                            self._mo_teis_einsum, 
                            self._norb)
                else:
                    qcSig_temp.apply_sqop(self._sq_ham)

                self._Hlanczos_vecs.append(qcSig_temp.get_state_deep())


        print(f"{'beta':>7}{'E(beta)':>18}{'N(params)':>14}{'N(CNOT)':>18}{'N(measure)':>20}")
        print('-------------------------------------------------------------------------------')
        print(f' {0.0:7.3f}    {self._Ekb[0]:+15.9f}    {self._n_classical_params:8}        {self._n_cnot:10}        {self._n_pauli_trm_measures:12}')

        if (self._print_summary_file):
            f = open(f"qite_{self._fname}_summary.dat", "w+", buffering=1)
            f.write(f"#{'beta':>7}{'E(beta)':>18}{'N(params)':>14}{'N(CNOT)':>18}{'N(measure)':>20}\n")
            f.write('#-------------------------------------------------------------------------------\n')
            f.write(f'  {0.0:7.3f}    {self._Ekb[0]:+15.9f}    {self._n_classical_params:8}        {self._n_cnot:10}        {self._n_pauli_trm_measures:12}\n')

        for kb in range(1, self._nbeta):
            if(self._use_exact_evolution):
                if(self._apply_ham_as_tensor):
                    self._qc.evolve_tensor_taylor(
                            self._nuclear_repulsion_energy, 
                            self._mo_oeis, 
                            self._mo_teis, 
                            self._mo_teis_einsum, 
                            self._norb,
                            self._db,
                            1.0e-15,
                            30,
                            True)

                    # print(f'norm before scaling: {self._qc.get_state().norm()}')

                    norm = 1.0 / self._qc.get_state().norm()
                    self._qc.scale(norm)

                    # print(f'norm after scaling: {self._qc.get_state().norm()}')

                    self._Ekb.append(np.real(self._qc.get_exp_val_tensor(
                            self._nuclear_repulsion_energy, 
                            self._mo_oeis, 
                            self._mo_teis, 
                            self._mo_teis_einsum, 
                            self._norb)))
                else:
                    self._qc.evolve_op_taylor(
                            self._sq_ham,
                            self._db,
                            1.0e-15,
                            30,
                            True)

                    # print(f'norm before scaling: {self._qc.get_state().norm()}')

                    norm = 1.0 / self._qc.get_state().norm()
                    self._qc.scale(norm)

                    # print(f'norm after scaling: {self._qc.get_state().norm()}')

                    self._Ekb.append(np.real(self._qc.get_exp_val(self._sq_ham)))

            else:
                self.do_qite_step()

                if(self._do_lanczos):
                    if(self._realistic_lanczos):
                        c_kb = np.exp(-2.0 * self._db * (self._Ekb[kb] - self._Ekb[0]))
                        self._c_list.append(c_kb)

                    else:
                        if(kb % self._lanczos_gap == 0):
                            self._lanczos_vecs.append(self._qc.get_state_deep())
                            qcSig_temp = qf.FCIComputer(self._nel, self._sz, self._norb)
                            qcSig_temp.set_state(self._qc.get_state_deep())

                            if(self._apply_ham_as_tensor):
                                qcSig_temp.apply_tensor_spat_012bdy(
                                    self._nuclear_repulsion_energy, 
                                    self._mo_oeis, 
                                    self._mo_teis, 
                                    self._mo_teis_einsum, 
                                    self._norb)
                            else:
                                qcSig_temp.apply_sqop(self._sq_ham)

                            self._Hlanczos_vecs.append(qcSig_temp.get_state_deep())

            print(f' {kb*self._db:7.3f}    {self._Ekb[kb]:+15.9f}    {self._n_classical_params:8}        {self._n_cnot:10}        {self._n_pauli_trm_measures:12}')
            if (self._print_summary_file):
                f.write(f'  {kb*self._db:7.3f}    {self._Ekb[kb]:+15.9f}    {self._n_classical_params:8}        {self._n_cnot:10}        {self._n_pauli_trm_measures:12}\n')
        self._Egs = self._Ekb[-1]

        if (self._print_summary_file):
            f.close()

    def print_expansion_ops(self):
        print('\nQITE expansion operators:')
        print('-------------------------')
        print(self._sig.str())
