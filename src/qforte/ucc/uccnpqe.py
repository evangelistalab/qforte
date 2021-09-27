"""
UCCNPQE classes
====================================
Classes for solving the schrodinger equation via measurement of its projections
and subsequent updates of the disentangled UCC amplitudes.
"""

import qforte
from qforte.abc.uccpqeabc import UCCPQE

from qforte.experiment import *
from qforte.maths import optimizer
from qforte.utils.transforms import *
from qforte.utils.state_prep import *
from qforte.utils.trotterization import trotterize

from qforte.helper.printing import matprint

import numpy as np
from scipy.linalg import lstsq
from collections import Counter

class UCCNPQE(UCCPQE):
    """
    A class that encompasses the three components of using the projective
    quantum eigensolver to optimize a disentangld UCCN-like wave function.

    UCC-PQE: (1) prepares a quantum state on the quantum computer
    representing the wave function to be simulated, (2) evaluates the residuals

    .. math::
        r_\mu = \langle \Phi_\mu | \hat{U}^\dagger(\mathbf{t}) \hat{H} \hat{U}(\mathbf{t}) | \Phi_0 \\rangle

    and (3) optimizes the wave fuction via projective solution of
    the UCC Schrodinger Equation via a quazi-Newton update equation.
    Using this strategy, an amplitude :math:`t_\mu^{(k+1)}` for iteration :math:`k+1`
    is given by

    .. math::
        t_\mu^{(k+1)} = t_\mu^{(k)} + \\frac{r_\mu^{(k)}}{\Delta_\mu}

    where :math:`\Delta_\mu` is the standard Moller Plesset denominator.

    Attributes
    ----------
    _ref : list
        The set of 1s and 0s indicating the initial quantum state.

    """
    def run(self,
            pool_type='SD',
            opt_thresh = 1.0e-5,
            opt_maxiter = 40,
            noise_factor = 0.0):

        if(self._state_prep_type != 'occupation_list'):
            raise ValueError("PQE implementation can only handle occupation_list Hartree-Fock reference.")

        self._pool_type = pool_type
        self._opt_thresh = opt_thresh
        self._opt_maxiter = opt_maxiter
        self._noise_factor = noise_factor

        self._tops = []
        self._tamps = []
        self._converged = 0

        self._res_vec_evals = 0
        self._res_m_evals = 0
        # list: tuple(excited determinant, phase_factor)
        self._excited_dets = []

        self._n_classical_params = 0
        self._n_cnot = 0
        self._n_pauli_trm_measures = 0
        # self._results = [] #keep for future implementations

        self.print_options_banner()
        self.fill_pool()

        self.initialize_ansatz()

        if(self._verbose):
            print('\nt operators included from pool: \n', self._tops)
            print('Initial tamplitudes for tops: \n', self._tamps)

        self.fill_excited_dets()
        self.build_orb_energies()
        self.solve()

        if(self._verbose):
            print('\nt operators included from pool: \n', self._tops)

            print('Final tamplitudes for tops:')
            print('------------------------------')
            for i, tamp in enumerate( self._tamps ):
                print(f'  {i:4}      {tamp:+12.8f}')

        self._n_nonzero_params = 0
        for tmu in self._tamps:
            if(np.abs(tmu) > 1.0e-12):
                self._n_nonzero_params += 1

        self._n_pauli_trm_measures = int(2*self._Nl*self._res_vec_evals*self._n_nonzero_params + self._Nl*self._res_vec_evals)

        self.print_summary_banner()
        self.verify_run()

    def run_realistic(self):
        raise NotImplementedError('run_realistic() is not fully implemented for UCCN-PQE.')

    def verify_run(self):
        self.verify_required_attributes()
        self.verify_required_PQE_attributes()
        self.verify_required_UCCPQE_attributes()

    def print_options_banner(self):
        print('\n-----------------------------------------------------')
        print('           Unitary Coupled Cluster PQE   ')
        print('-----------------------------------------------------')

        print('\n\n                 ==> UCC-PQE options <==')
        print('---------------------------------------------------------')
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

        res_thrsh_str = '{:.2e}'.format(self._opt_thresh)
        print('DIIS maxiter:                            ',  self._opt_maxiter)
        print('DIIS res-norm threshold:                 ',  res_thrsh_str)

        print('Operator pool type:                      ',  str(self._pool_type))


    def print_summary_banner(self):

        print('\n\n                   ==> UCC-PQE summary <==')
        print('-----------------------------------------------------------')
        print('Final UCCN-PQE Energy:                      ', round(self._Egs, 10))
        print('Number of operators in pool:                 ', len(self._qubit_pool))
        print('Final number of amplitudes in ansatz:        ', len(self._tamps))
        print('Number of classical parameters used:         ', len(self._tamps))
        print('Number of non-zero parameters used:          ', self._n_nonzero_params)
        print('Number of CNOT gates in deepest circuit:     ', self._n_cnot)
        print('Number of Pauli term measurements:           ', self._n_pauli_trm_measures)
        print('Number of residual vector evaluations:       ', self._res_vec_evals)
        print('Number of residual element evaluations*:     ', self._res_m_evals)
        print('Number of non-zero res element evaluations:  ', int(self._res_vec_evals)*self._n_nonzero_params)

    def solve(self):
        self.diis_solve(self.get_residual_vector)

    def fill_excited_dets(self):

        reference_state = qforte.QubitBasis(self._nqb)
        for k, occ in enumerate(self._ref):
            reference_state.set_bit(k, occ)

        self._excited_dets = [operator_to_determinant(qubit_operator, reference_state) for _, qubit_operator in self._qubit_pool]

    def get_residual_vector(self, trial_amps):
        """Returns the residual vector with elements pertaining to all operators
        in the ansatz circuit.

        Parameters
        ----------
        trial_amps : list of floats
            The list of (real) floating point numbers which will characterize
            the state preparation circuit used in calculation of the residuals.
        """
        if(self._pool_type == 'sa_SD'):
            raise ValueError('Must use single term particle-hole nbody operators for residual calculation')

        U = self.ansatz_circuit(trial_amps)

        qc_res = qforte.Computer(self._nqb)
        qc_res.apply_circuit(self._Uprep)
        qc_res.apply_circuit(U)
        qc_res.apply_operator(self._qb_ham)
        qc_res.apply_circuit(U.adjoint())

        coeffs = qc_res.get_coeff_vec()
        residuals = []

        for I, phase_factor in self._excited_dets:

            # Get the residual element, after accounting for numerical noise.
            res_m = coeffs[I] * phase_factor
            if(np.imag(res_m) != 0.0):
                raise ValueError("residual has imaginary component, something went wrong!!")

            if(self._noise_factor > 1e-12):
                res_m = np.random.normal(np.real(res_m), self._noise_factor)

            residuals.append(res_m)

        self._res_vec_evals += 1
        self._res_m_evals += len(self._tamps)

        return residuals

    def initialize_ansatz(self):
        """Adds all operators in the pool to the list of operators in the circuit,
        with amplitude 0.
        """
        for l in range(len(self._qubit_pool)):
            self._tops.append(l)
            self._tamps.append(0.0)

UCCNPQE.diis_solve = optimizer.diis_solve
