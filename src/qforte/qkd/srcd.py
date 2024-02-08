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
            s=3,
            dt=0.5,
            target_root=0,
            use_exact_evolution=False,
            diagonalize_each_step=True
            ):

        self._s = s
        self._nstates = s+1
        self._dt = dt
        self._target_root = target_root
        self._use_exact_evolution = use_exact_evolution
        self._diagonalize_each_step = diagonalize_each_step

        self._n_classical_params = 0
        self._n_cnot = 0
        self._n_pauli_trm_measures = 0

        # Print options banner (should done for all algorithms).
        self.print_options_banner()

        self.common_run()

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
        # print('Trotter order (rho):                     ',  self._trotter_order)
        # print('Trotter number (m):                      ',  self._trotter_number)
        # print('Use exact time evolution?:               ',  self._use_exact_evolution)
        # print('Use fast version of algorithm:           ',  str(self._fast))
        # if(self._fast):
        #     print('Measurement varience thresh:             ',  'NA')
        # else:
        #     print('Measurement varience thresh:             ',  0.01)

        # Specific SRCD options.
        print('Dimension of Krylov space (N):           ',  self._nstates)
        # print('Delta t (in a.u.):                       ',  self._dt)
        print('Target root:                             ',  str(self._target_root))


    def print_summary_banner(self):
        cs_str = '{:.2e}'.format(self._Scond)

        print('\n\n                     ==> CD summary <==')
        print('-----------------------------------------------------------')
        print('Condition number of overlap mat k(S):      ', cs_str)
        print('Final SRCD ground state Energy:           ', round(self._Egs, 10))
        print('Final SRCD target state Energy:           ', round(self._Ets, 10))
        print('Number of classical parameters used:       ', self._n_classical_params)
        print('Number of CNOT gates in deepest circuit:   ', self._n_cnot)
        print('Number of Pauli term measurements:         ', self._n_pauli_trm_measures)

    def build_qk_mats(self):
        self.build_cd_mats()

    # def build_qk_mats_fast(self):

    def build_cd_mats(self):
        if(self._computer_type == 'fock'):
            if(self._trotter_order != 1):
                raise ValueError("fock computer SRCD not currently implemented")
            
            # return self.build_cd_mats_fock()
        
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

        h_mat = np.zeros((self._nstates,self._nstates), dtype=complex)
        s_mat = np.zeros((self._nstates,self._nstates), dtype=complex)

        # Store these vectors for the aid of MRSQK
        self._omega_lst = []
        Homega_lst = []

        hermitian_pairs = qforte.SQOpPool()

        # this is updated, evolution time is now just 1.0 here
        hermitian_pairs.add_hermitian_pairs(1.0, self._sq_ham)

        QC = qforte.FCIComputer(
                self._nel, 
                self._2_spin, 
                self._norb)
            
        QC.hartree_fock()

        if(self._diagonalize_each_step):
            print('\n\n')

            print(f"{'k(S)':>7}{'E(Npar)':>19}{'N(params)':>14}{'N(CNOT)':>18}{'N(measure)':>20}")
            print('-------------------------------------------------------------------------------')

            if (self._print_summary_file):
                f = open("summary.dat", "w+", buffering=1)
                f.write(f"#{'k(S)':>7}{'E(Npar)':>19}{'N(params)':>14}{'N(CNOT)':>18}{'N(measure)':>20}\n")
                f.write('#-------------------------------------------------------------------------------\n')

    
        """In reviewing this there is going to be an inherent ordering probelm. I want to apply 
        based on hermitian paris of SQ operators but the qb hamiltonain has been 'simplified' 
        and looses the exact correspondance to the sq hamiltonain"""
        for m in range(self._nstates):

            if(m>0):
                # Compute U_m |φ>
                if(self._use_exact_evolution):
                    QC.evolve_op_taylor(
                        self._sq_ham,
                        self._dt,
                        1.0e-15,
                        30)

                else:
                    QC.evolve_pool_trotter(
                        hermitian_pairs,
                        self._dt,
                        self._trotter_number,
                        self._trotter_order,
                        antiherm=False,
                        adjoint=False)

            C = QC.get_state_deep()
         
            self._omega_lst.append(C)

            QC.apply_sqop(self._sq_ham)

            Sig = QC.get_state_deep()

            Homega_lst.append(Sig)

            # Compute S_mn = <φ| U_m^\dagger U_n |φ> and H_mn = <φ| U_m^\dagger H U_n |φ>
            for n in range(len(self._omega_lst)):
                h_mat[m][n] = self._omega_lst[m].vector_dot(Homega_lst[n])
                h_mat[n][m] = np.conj(h_mat[m][n])
                s_mat[m][n] = self._omega_lst[m].vector_dot(self._omega_lst[n])
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
                # self._n_cnot = 2 * Um.get_num_cnots()
                self._n_cnot = 0
                self._n_pauli_trm_measures  = k * self._Nl
                self._n_pauli_trm_measures += k * (k-1) * self._Nl
                self._n_pauli_trm_measures += k * (k-1)

                print(f' {scond:7.2e}    {np.real(evals[self._target_root]):+15.9f}    {self._n_classical_params:8}        {self._n_cnot:10}        {self._n_pauli_trm_measures:12}')
                if (self._print_summary_file):
                    f.write(f'  {scond:7.2e}    {np.real(evals[self._target_root]):+15.9f}    {self._n_classical_params:8}        {self._n_cnot:10}        {self._n_pauli_trm_measures:12}\n')

        if (self._diagonalize_each_step and self._print_summary_file):
            f.close()

        # Some of these doen't matter for classical Davidson
        self._n_classical_params = self._nstates
        # self._n_cnot = 2 * Um.get_num_cnots()
        self._n_cnot = 0
        # diagonal terms of Hbar
        self._n_pauli_trm_measures  = self._nstates * self._Nl
        # off-diagonal of Hbar (<X> and <Y> of Hadamard test)
        self._n_pauli_trm_measures += self._nstates*(self._nstates-1) * self._Nl
        # off-diagonal of S (<X> and <Y> of Hadamard test)
        self._n_pauli_trm_measures += self._nstates*(self._nstates-1)


        return s_mat, h_mat


    