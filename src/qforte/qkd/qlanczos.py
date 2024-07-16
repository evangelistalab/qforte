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

from qforte.ite import QITE

from qforte.maths.eigsolve import canonical_geig_solve

from qforte.utils.state_prep import *
from qforte.utils.trotterization import (trotterize,
                                         trotterize_w_cRz)

import numpy as np

class QLANCZOS(QSD):
    """A quantum subspace diagonalization algorithm that generates the many-body
    basis from different durations of imaginary time evolution propagated by QITE:

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

            # old srqk options

            # s=3,
            # dt=0.5,
            target_root=0,
            # use_exact_evolution=False,
            diagonalize_each_step=True, 

            # # put qite and qite.run options

            beta=1.0,
            db=0.2,
            expansion_type='SD',
            use_exact_evolution=False,
            sparseSb=False,
            low_memorySb=False,
            second_order=True,
            b_thresh=1.0e-6,
            x_thresh=1.0e-10,
            do_lanczos=True,
            lanczos_gap=2,
            realistic_lanczos=True,
            fname=None 

            # new ql options

            ):


        # OLD STUFF
        # self._s = s
        # self._nstates = s+1
        # self._dt = dt
        self._target_root = target_root
        # self._use_exact_evolution = use_exact_evolution
        self._diagonalize_each_step = diagonalize_each_step

        self._n_classical_params = 0
        self._n_cnot = 0
        self._n_pauli_trm_measures = 0

        ###QITE STUFF
        self._beta = beta
        self._db = db
        self._use_exact_evolution = use_exact_evolution
        self._expansion_type = expansion_type
        self._sparseSb = sparseSb
        self._low_memorySb = low_memorySb
        self._second_order = second_order
        self._b_thresh = b_thresh
        self._x_thresh = x_thresh
        self._do_lanczos = do_lanczos
        self._lanczos_gap = lanczos_gap
        self._realistic_lanczos = realistic_lanczos
        self._fname = fname

        if(self._do_lanczos != True):
            print(f'cannot run QLanczos with do_lanczos option set to {self._do_lanczos}, setting option to True')
            self._do_lanczos = True

        if(self._use_exact_evolution != False):
            print(f'cannot run QLanczos with use_exact_evolution option set to {self._use_exact_evolution}, setting option to False')
            self._use_exact_evolution = False

        if(self._lanczos_gap % 2 != 0):
            raise ValueError('Lanczos gap must be even to include HF state')

        # Print options banner (should done for all algorithms).
        self.print_options_banner()

        print('\nNow running QITE subroutine')

        self._qite = QITE(self._sys, 
                          self._computer_type, 
                          self._apply_ham_as_tensor, 
                          self._ref, 
                          self._state_prep_type, 
                          self._trotter_order, 
                          self._trotter_number, 
                          self._fast, 
                          self._verbose, 
                          self._print_summary_file)
                          
        self._qite.run(self._beta, 
                       self._db,
                       self._use_exact_evolution,
                       self._expansion_type, 
                       self._sparseSb, 
                       self._low_memorySb, 
                       self._second_order, 
                       self._b_thresh, 
                       self._x_thresh, 
                       self._do_lanczos, 
                       self._lanczos_gap,
                       self._realistic_lanczos, 
                       self._fname)

        self.common_run()

    # Define Algorithm abstract methods.
    def run_realistic(self):
        raise NotImplementedError('run_realistic() is not fully implemented for SRQK.')

    def verify_run(self):
        self.verify_required_attributes()
        self.verify_required_QSD_attributes()

# you gotta fix this at some point
    def print_options_banner(self):
        print('\n\n-----------------------------------------------------')
        print('         Quantum Imaginary Time Lanczos   ')
        print('-----------------------------------------------------\n\n')

        print('\n\n                     ==> Lanczos options <==')
        print('-----------------------------------------------------------')
        # General algorithm options.
        print('Trial reference state:                   ',  ref_string(self._ref, self._nqb))
        print('Number of Hamiltonian Pauli terms:       ',  self._Nl)
        print('Trial state preparation method:          ',  self._state_prep_type)
        print('Trotter order (rho):                     ',  self._trotter_order)
        print('Trotter number (m):                      ',  self._trotter_number)
        # print('Use exact time evolution?:               ',  self._use_exact_evolution)
        print('Use fast version of algorithm:           ',  str(self._fast))
        if(self._fast):
            print('Measurement varience thresh:             ',  'NA')
        else:
            print('Measurement varience thresh:             ',  0.01)

        # Specific SRQK options.
        # print('Dimension of Krylov space (N):           ',  self._nstates)
        # print('Delta beta (in a.u.):                       ',  self._dt)
        print('Target root:                             ',  str(self._target_root))


    def print_summary_banner(self):
        cs_str = '{:.2e}'.format(self._Scond)

        print('\n\n                     ==> Lanczos summary <==')
        print('-----------------------------------------------------------')
        print('Condition number of overlap mat k(S):      ', cs_str)
        print('Final QLanczos ground state Energy:        ', round(self._Egs, 10))
        # print('Final SRQK target state Energy:           ', round(self._Ets, 10))
        print('Number of classical parameters used:       ', self._n_classical_params)
        print('Number of CNOT gates in deepest circuit:   ', self._n_cnot)
        print('Number of Pauli term measurements:         ', self._n_pauli_trm_measures)

    def build_qk_mats(self):
        if (self._fast):
            return self.build_qk_mats_fast()
        else:
            return self.run_realistic()


    def build_qk_mats_fast(self):
        if(self._computer_type == 'fock'):
            raise NotImplementedError('QLanczos is not implemented for fock computer type')
        
        elif(self._computer_type == 'fci'):
            return self.build_qk_mats_fast_fci()
        
        else:
            raise ValueError(f"{self._computer_type} is an unrecognized computer type.") 

    
    def build_qk_mats_fast_fci(self):
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

        # h_mat = np.zeros((self._nstates,self._nstates), dtype=complex)
        # s_mat = np.zeros((self._nstates,self._nstates), dtype=complex)

        if(self._realistic_lanczos):
            # t_dim = self._qite._t # total number of steps
            # print(f't_dim:{t_dim}')
            # # print(f'length of lanczos_vecs:{len(self._qite._lanczos_vecs)}')

            # tdim = self._nbeta - 1
            # if t_dim % 2 == 0:
            ndim = int((self._qite._nbeta) // self._lanczos_gap + 1)
            # else:
            #     n_dim = int((t+1)//2)

            # ndim = len(self._qite._c_list)
            print(f'n_dim:{ndim}')

            h_mat = np.zeros((ndim, ndim), dtype=complex)
            s_mat = np.zeros((ndim, ndim), dtype=complex)

            if(self._diagonalize_each_step):
                print('\n\n')

                print(f"{'Beta':>7}{'k(S)':>7}{'E(Npar)':>19}")
                print('-------------------------------------------------------------------------------')

                if (self._print_summary_file):
                    # put fname here too
                    f2 = open(f"{self._qite._fname}_realistic_{self._realistic_lanczos}_lanczos_summary.dat", "w+", buffering=1)
                    f2.write(f"#{'Beta':>7}{'k(S)':>7}{'E(Npar)':>19}\n")
                    f2.write('#-------------------------------------------------------------------------------\n')

            # build elements of S and H matrix
            # see quket qlanczos implementaion for example on index mapping
            for i in range(ndim):
                i_gap = self._lanczos_gap * i

                for j in range(i+1):
                    j_gap = self._lanczos_gap * j
                    r = (i_gap + j_gap) // self._lanczos_gap

                    n = 1
                    d = 1

                    for ix in range(j_gap+1, r+1):
                        n *= self._qite._c_list[ix]

                    for ix in range(r+1, i_gap+1):
                        d *= self._qite._c_list[ix]

                    # if i == j:
                    #     s_mat[i,j] = 1.0
                    # else:
                    #     s_mat[j,i] = s_mat[i,j] = np.sqrt(n / d)
                    s_mat[j,i] = s_mat[i,j] = np.sqrt(n / d)
                    h_mat[j,i] = h_mat[i,j] = s_mat[i,j] * self._qite._Ekb[r]

                if (self._diagonalize_each_step):
                    # print('\n')
                    # print(f'iteration: {i}')
                    # print('S MAT')
                    # matprint(s_mat)
                    # print('\n')
                    # print('H MAT')
                    # matprint(h_mat)
                    # print('\n')
                    k = i+1
                    evals, evecs = canonical_geig_solve(s_mat[0:k, 0:k],
                                    h_mat[0:k, 0:k],
                                    print_mats=False,
                                    sort_ret_vals=True)

                    scond = np.linalg.cond(s_mat[0:k, 0:k])

                    print(f'{i * self._lanczos_gap * self._db:7.3f} {scond:7.2e}    {np.real(evals[0]):+15.9f} ')

                    if (self._print_summary_file):
                        f2.write(f'{i * self._lanczos_gap * self._db:7.3f} {scond:7.2e}    {np.real(evals[0]):+15.9f} \n')

            if (self._diagonalize_each_step and self._print_summary_file):
                f2.close()

            return s_mat, h_mat

        else:
            n_lanczos_vecs = len(self._qite._lanczos_vecs)
            h_mat = np.zeros((n_lanczos_vecs, n_lanczos_vecs), dtype=complex)
            s_mat = np.zeros((n_lanczos_vecs, n_lanczos_vecs), dtype=complex)

            if(self._diagonalize_each_step):
                print('\n\n')

                print(f"{'Beta':>7}{'k(S)':>7}{'E(Npar)':>19}")
                print('-------------------------------------------------------------------------------')

                if (self._print_summary_file):
                    # put fname here too
                    f2 = open(f"{self._qite._fname}_realistic_{self._realistic_lanczos}_lanczos_summary.dat", "w+", buffering=1)
                    f2.write(f"#{'Beta':>7}{'k(S)':>7}{'E(Npar)':>19}\n")
                    f2.write('#-------------------------------------------------------------------------------\n')

            for m in range(n_lanczos_vecs):
                for n in range(m+1):
                    h_mat[m][n] = self._qite._lanczos_vecs[m].vector_dot(self._qite._Hlanczos_vecs[n])
                    h_mat[n][m] = np.conj(h_mat[m][n])
                    s_mat[m][n] = self._qite._lanczos_vecs[m].vector_dot(self._qite._lanczos_vecs[n])
                    s_mat[n][m] = np.conj(s_mat[m][n])

                if (self._diagonalize_each_step):
                    # print('\n')
                    # print(f'iteration: {m}')
                    # print('S MAT')
                    # matprint(s_mat)
                    # print('\n')
                    # print('H MAT')
                    # matprint(h_mat)
                    # print('\n')
                    k = m+1
                    evals, evecs = canonical_geig_solve(s_mat[0:k, 0:k],
                                    h_mat[0:k, 0:k],
                                    print_mats=False,
                                    sort_ret_vals=True)

                    scond = np.linalg.cond(s_mat[0:k, 0:k])

                    print(f'{m * self._lanczos_gap * self._db:7.3f} {scond:7.2e}    {np.real(evals[0]):+15.9f} ')

                    if (self._print_summary_file):
                        f2.write(f'{m * self._lanczos_gap * self._db:7.3f} {scond:7.2e}    {np.real(evals[0]):+15.9f} \n')

            if (self._diagonalize_each_step and self._print_summary_file):
                f2.close()

            return s_mat, h_mat
