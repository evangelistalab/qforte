"""
SRCD classes
=================================================
Classes for calculating reference states for quantum
mechanical systems for the single referece classical
Davidson algorithm.
"""

import qforte
from qforte.abc.qsdabc import QSD
from qforte.helper.printing import matprint

from qforte.maths.eigsolve import canonical_geig_solve
from qforte.maths import gram_schmidt

from qforte.utils.state_prep import *
from qforte.utils.trotterization import (trotterize,
                                         trotterize_w_cRz)

import numpy as np

class SRCD(QSD):
    """A quantum subspace diagonalization algorithm that generates the many-body
    basis from different durations of real time evolution:

    .. math::
        | \Psi_n \\rangle = e^{-i n \Delta t \hat{H}} | \Phi_0 \\rangle

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
    def run(self,
            target_root=0,
            max_itr=50,
            thresh=1e-9,
            use_exact_evolution=False,
            diagonalize_each_step=True
            ):

        

        self.max_itr = max_itr
        self.thresh = thresh
        self._target_root = target_root
        self._use_exact_evolution = use_exact_evolution
        self._diagonalize_each_step = diagonalize_each_step

        self._n_classical_params = 0
        self._n_cnot = 0
        self._n_pauli_trm_measures = 0

        # Print options banner (should done for all algorithms).
        self.print_options_banner()

        self._timer = qforte.local_timer()
        self._timer.reset()
        self.common_run()
        self._timer.record("Run")

        # Temporary solution, replacing Egs with lambda_low
        self._Egs = self._lambda_low

    # Define Algorithm abstract methods.
    def run_realistic(self):
        raise NotImplementedError('run_realistic() is not fully implemented for SRCD.')

    def verify_run(self):
        self.verify_required_attributes()
        self.verify_required_QSD_attributes()

    def print_options_banner(self):
        print('\n-----------------------------------------------------')
        print('           Single Reference Classical Davidson   ')
        print('-----------------------------------------------------')

        print('\n\n                     ==> CD options <==')
        print('-----------------------------------------------------------')
        # General algorithm options.
        print('Trial reference state:                   ',  ref_string(self._ref, self._nqb))
        print('Number of Hamiltonian Pauli terms:       ',  self._Nl)
        print('Trial state preparation method:          ',  self._state_prep_type)

        # Specific SRCD options.
        print('Target root:                             ',  str(self._target_root))


    def print_summary_banner(self):
        cs_str = '{:.2e}'.format(self._Scond)

        print('\n\n                     ==> CD summary <==')
        print('-----------------------------------------------------------')
        print('Condition number of overlap mat k(S):      ', cs_str)
        print('Final SRCD ground state Energy:           ', round(self._lambda_low, 10))
        print('Final SRCD target state Energy:           ', round(self._Ets, 10))
        print('Number of classical parameters used:       ', self._n_classical_params)
        print('Number of CNOT gates in deepest circuit:   ', self._n_cnot)
        print('Number of Pauli term measurements:         ', self._n_pauli_trm_measures)

        print("\n\n")
        print(self._timer)

    def build_qk_mats(self):
        return self.build_cd_mats()

    def build_cd_mats(self):
        if(self._computer_type == 'fock'):
            raise ValueError("fock computer SRCD not currently implemented")
        
        elif(self._computer_type == 'fci'):
            if(self._trotter_order not in [1, 2]):
                raise ValueError("fci computer SRCD only compatible with 1st and 2nd order trotter currently")
            
            return self.build_cd_mats_fci()
        
        else:
            raise ValueError(f"{self._computer_type} is an unrecognized computer type.") 
    
    # TODO(Victor): This is where you should implement your davidson algorithm,
    # for now, try actuyally construction the subspace H and S matricies 
    # (S should be the identity for davidson as long as you orthogonalize)
    def build_cd_mats_fci(self):
        """Returns matrices S and H needed for the CD algorithm 

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

        if(self._diagonalize_each_step):
            print('\n\n')

            print(f"{'k(S)':>7}{'E(Npar)':>19}{'dE(Npar)':>19}{'N(params)':>14}")
            print('--------------------------------------------------------------------------')

            if (self._print_summary_file):
                f = open("summary.dat", "w+", buffering=1)
                f.write(f"#{'k(S)':>7}{'E(Npar)':>19}{'dE(Npar)':>19}{'N(params)':>14}\n")
                f.write('#-------------------------------------------------------------------------------\n')

        # Store vectors for subspace construction
        self._omega_lst = []

        QC = qforte.FCIComputer(
                self._nel, 
                self._2_spin, 
                self._norb)
        
        PSI = QC.get_state_deep() # empty tensor for zaxpy routine, same dimension as element of guess space

        QC.hartree_fock()

        C_0 = QC.get_state_deep() # our initial guess vector is a tensor with shape (1 x n)

        for m in range(1, self.max_itr):

            if(m<=1):

                self._omega_lst.append(C_0)
                lambda_old = 1
                lambda_low = 0.0

            lambda_old = lambda_low
            self._omega_lst = gram_schmidt.orthogonalize(self._omega_lst)

            Homega_lst = []

            for C in self._omega_lst:

                QC.set_state(C)

                if(self._apply_ham_as_tensor):
                    QC.apply_tensor_spat_012bdy(
                        self._nuclear_repulsion_energy, 
                        self._mo_oeis, 
                        self._mo_teis, 
                        self._mo_teis_einsum, 
                        self._norb)
                else:   
                    QC.apply_sqop(self._sq_ham)

                Sig = QC.get_state_deep()

                Homega_lst.append(Sig)
        
            # utilizing h_mat as the subspace matrix to be diagonalized until convergence threshold is reached 
            h_mat = np.zeros((len(self._omega_lst), len(self._omega_lst)), dtype=complex)
            s_mat = np.eye(len(self._omega_lst), dtype=complex)

            for m in range(len(self._omega_lst)):
                for n in range(len(Homega_lst)):

                    h_mat[m][n] = self._omega_lst[m].vector_dot(Homega_lst[n])
                    s_mat[m][n] = self._omega_lst[m].vector_dot(self._omega_lst[n])

            # get λ, v of T
            evals, evecs = np.linalg.eig(h_mat)

            idx = evals.argsort()
            sorted_eigenvals = evals[idx]
            sorted_eigenvecs = evecs[:,idx]
            lambda_low = sorted_eigenvals[0]
            v_low = sorted_eigenvecs[:, 0]

            # |Ψ> = V⋅v
            for a, v in zip(v_low, self._omega_lst):

                PSI.zaxpy(x = v, alpha = a)

            # r = [λ - H_diag]⁻¹[H - λ]|Ψ>
            QC.set_state(PSI)

            psi_2 = QC.get_state_deep()

            if(self._apply_ham_as_tensor):
                QC.apply_tensor_spat_012bdy(
                    self._nuclear_repulsion_energy, 
                    self._mo_oeis, 
                    self._mo_teis, 
                    self._mo_teis_einsum, 
                    self._norb)
            else:   
                QC.apply_sqop(self._sq_ham)

            psi_1 = QC.get_state_deep()

            psi_2.scale(-lambda_low)

            psi_1.add(psi_2)

            QC.set_state(psi_1)

            temp_sqop = qforte.SQOperator()

            temp_sqop.add_op(self._sq_ham)

            temp_sqop.mult_coeffs(-1)

            temp_sqop.add_term(lambda_low, [], [])

            QC.apply_diagonal_of_sqop(temp_sqop, invert_coeff=True)

            R_vec = QC.get_state_deep()

            self._omega_lst.append(R_vec)

            if (self._diagonalize_each_step):
                # TODO (cleanup): have this print to a separate file
                k = m+1
                evals, evecs = canonical_geig_solve(s_mat[0:k, 0:k],
                                   h_mat[0:k, 0:k],
                                   print_mats=False,
                                   sort_ret_vals=True)

                scond = np.linalg.cond(s_mat[0:k, 0:k])
                self._n_classical_params = k
                # self._n_cnot = 2 * Um.get_num_cnots()
                self._n_cnot = 0
                self._n_pauli_trm_measures  = k * self._Nl
                self._n_pauli_trm_measures += k * (k-1) * self._Nl
                self._n_pauli_trm_measures += k * (k-1)

                delta_e = np.real(lambda_low - lambda_old)

                print(f' {scond:7.2e}    {np.real(lambda_low):15.9f}    {np.real(delta_e):15.9f}    {self._n_classical_params:8}')
                # if (self._print_summary_file):
                    # f.write(f'  {scond:7.2e}    {np.real(evals[self._target_root]):+15.9f}    {self._n_classical_params:8}        {self._n_cnot:10}        {self._n_pauli_trm_measures:12}\n')


            if abs(lambda_old - lambda_low) <= self.thresh:
                print('\n Davidson Converged!')
                self._Egs = np.real(np.real(lambda_low))
                self._lambda_low = np.real(lambda_low)
                break

        print(f'davidson iteration: {m + 1}, energy: {np.real(lambda_low):15.9f}, energy difference: {lambda_low - lambda_old}')

        return s_mat, h_mat