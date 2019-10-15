"""
A class for using an experiment to execute the variational quantum eigensolver
"""

import qforte
# from qforte.experiment import *
# from qforte.utils.transforms import *
# from qforte.ucc.ucc_helpers import *
from qforte.rtl import rtl_helpers
from qforte.rtl import artl_helpers
import numpy as np
import scipy
from scipy.optimize import minimize

class RTLVQE(object):
    """
    RTLVQE is a class that encompases the three componants of using the variational
    quantum eigensolver to optemize the time step for each reference in RTQL. Over a number
    of iterations the VQE: (1) prepares a quantum state on the quantum computer
    representing the wave function to be simulated, (2) evauates the energy by
    measurement, and (3) optemizes the the wave funciton by minimizing the energy
    via a gradient free algorithm such as nelder-mead simplex algroithm.
    VQE object constructor.
    :param n_qubits: the number of qubits for the quantum experiment.
    :param operator: the qubit operator to be measured.
    :param N_samples: the number of measurements made for each term in the operator
    :param many_preps: do a state preparation for every measurement (like a physical
        quantum computer would need to do).
    """


    def __init__(self, ref_lst, Nepr, operator, fast=True,
                    N_samples = 100, many_preps = False,
                    optimizer='nelder-mead'):

        self._ref_lst = ref_lst
        self._Nrefs = len(ref_lst)
        self._nstates_per_ref = Nepr
        self._operator = operator
        self._fast = fast
        self._N_samples = N_samples
        self._many_preps = many_preps
        self._optimizer = optimizer


    """
    Optemizes the params of the generator by minimization of the
    'experimental_avg' function.
    """


    def expecation_val_func_fast(self, x):

        #NOTE: need get nqubits from Molecule class attribute instead of ref list length
        # Also true for UCC functions
        num_refs = self._Nrefs
        num_tot_basis = num_refs * self._nstates_per_ref
        nqubits = len(self._ref_lst[0])

        h_mat = np.zeros((num_tot_basis,num_tot_basis), dtype=complex)
        s_mat = np.zeros((num_tot_basis,num_tot_basis), dtype=complex)

        dt_lst = x

        if(self._fast):
            print('using faster fast algorithm lol')
            s_mat, h_mat = rtl_helpers.get_mr_mats_fast(self._ref_lst, self._nstates_per_ref,
                                                        dt_lst, self._operator,
                                                        nqubits)

        else:
            for I in range(num_refs):
                for J in range(num_refs):
                    for m in range(self._nstates_per_ref):
                        for n in range(self._nstates_per_ref):
                            p = I*self._nstates_per_ref + m
                            q = J*self._nstates_per_ref + n
                            if(q>=p):
                                ref_I = self._ref_lst[I]
                                ref_J = self._ref_lst[J]
                                dt_I = dt_lst[I]
                                dt_J = dt_lst[J]

                                h_mat[p][q] = rtl_helpers.mr_matrix_element(ref_I, ref_J, dt_I, dt_J,
                                                                            m, n, self._operator,
                                                                            nqubits, self._operator)
                                h_mat[q][p] = np.conj(h_mat[p][q])

                                s_mat[p][q] = rtl_helpers.mr_matrix_element(ref_I, ref_J, dt_I, dt_J,
                                                                            m, n, self._operator,
                                                                            nqubits)
                                s_mat[q][p] = np.conj(s_mat[p][q])



        # if(print_mats):
        #     print('------------------------------------------------')
        #     print('   Matricies for MR Quantum Real-Time Lanczos')
        #     print('------------------------------------------------')
        #     print('self._Nrefs:             ', num_refs)
        #     print('Nevos per ref:     ', self._nstates_per_ref)
        #     print('Ntot states  :     ', num_tot_basis)
        #     print('Delta t list :     ', dt_lst)
        #
        #     print("\nS:\n")
        #     rtl_helpers.matprint(s_mat)
        #
        #     print('\nk(S): ', np.linalg.cond(s_mat))
        #
        #     print("\nHbar:\n")
        #     rtl_helpers.matprint(h_mat)
        #
        #     print('\nk(Hbar): ', np.linalg.cond(h_mat))

        evals, evecs = rtl_helpers.canonical_geig_solve(s_mat, h_mat)
        print('\nRTLanczos (unsorted!) evals from measuring ancilla:\n', evals)

        evals_sorted = np.sort(evals)

        if(np.abs(np.imag(evals_sorted[0])) < 1.0e-3):
            Eo = np.real(evals_sorted[0])
        elif(np.abs(np.imag(evals_sorted[1])) < 1.0e-3):
            print('Warning: problem may be ill condidtiond, evals have imaginary components')
            Eo = np.real(evals_sorted[1])
        else:
            print('Warding: problem may be extremely ill conditioned, check evals and k(S)')
            Eo = 0.0

        # if(return_all_eigs or return_S or return_Hbar):
        #     return_list = [Eo]
        #     if(return_all_eigs):
        #         return_list.append(evals_sorted)
        #     if(return_S):
        #         return_list.append(s_mat)
        #     if(return_Hbar):
        #         return_list.append(h_mat)
        #
        #     return return_list

        return Eo

    def do_vqe(self, initial_dt, maxiter=20000):

        x0 = []
        for i in range(self._Nrefs):
            x0.append(initial_dt)

        # Set options dict
        opts = {}
        opts['xtol'] = 1e-3
        opts['ftol'] = 1e-4
        opts['disp'] = True
        opts['maxiter'] = maxiter
        # opts['maxfev'] = 1

        self._inital_guess_energy = self.expecation_val_func_fast(x0)
        self._result = minimize(self.expecation_val_func_fast, x0,
                                method=self._optimizer,
                                options=opts)

    def get_result(self):
        return self._result

    def get_energy(self):
        if(self._result.success):
            return self._result.fun
        else:
            print('\nresult:', self._result)
            # raise ValueError('Minimum energy invalid, optemization did not converge')
            return self._result.fun

    def get_inital_guess_energy(self):
        return self._inital_guess_energy
