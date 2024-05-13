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
from qforte.utils import moment_energy_corrections

from qforte.helper.printing import matprint

import numpy as np
from scipy.linalg import lstsq

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
            noise_factor = 0.0,
            optimizer = 'jacobi'):

        if(self._state_prep_type != 'occupation_list'):
            raise ValueError("PQE implementation can only handle occupation_list Hartree-Fock reference.")

        self._pool_type = pool_type
        self._optimizer = optimizer
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
        self._excited_dets_fci_comp = []

        self._n_classical_params = 0
        self._n_cnot = 0
        self._n_pauli_trm_measures = 0
        # self._results = [] #keep for future implementations

        self.print_options_banner()

        self._timer = qforte.local_timer()

        self._timer.reset()
        self.fill_pool()
        self._timer.record("fill_pool")


        if self._verbose:
            print('\n\n-------------------------------------')
            print('   Second Quantized Operator Pool')
            print('-------------------------------------')
            print(self._pool_obj.str())

        self._timer.reset()
        self.initialize_ansatz()
        self._timer.record("initialize_ansatz")

        if(self._verbose):
            print('\nt operators included from pool: \n', self._tops)
            print('Initial tamplitudes for tops: \n', self._tamps)

        self._timer.reset()
        self.fill_excited_dets()
        self._timer.record("fill_excited_dets")

        self._timer.reset()
        self.build_orb_energies()
        self._timer.record("build_orb_energies")
        
        self._timer.reset()
        self.solve()
        self._timer.record("solve")

        if self._max_moment_rank:
            print('\nConstructing Moller-Plesset and Epstein-Nesbet denominators')
            self.construct_moment_space()
            print('\nComputing non-iterative energy corrections')
            self.compute_moment_energies()

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
        print('Number of Electrons:                     ',  self._nel)
        print('Multiplicity:                            ',  self._mult)
        print('Number spatial orbitals:                 ',  self._norb)
        print('Number of Hamiltonian Pauli terms:       ',  self._Nl)
        print('Trial state preparation method:          ',  self._state_prep_type)
        print('Trotter order (rho):                     ',  self._trotter_order)
        print('Trotter number (m):                      ',  self._trotter_number)
        print('Use fast version of algorithm:           ',  str(self._fast))
        if(self._fast):
            print('Measurement varience thresh:             ',  'NA')
        else:
            print('Measurement varience thresh:             ',  0.01)

        print('Use qubit excitations:                   ', self._qubit_excitations)
        print('Use compact excitation circuits:         ', self._compact_excitations)

        res_thrsh_str = '{:.2e}'.format(self._opt_thresh)
        print('Optimizer:                               ', self._optimizer)
        if self._diis_max_dim >= 2 and self._optimizer.lower() == 'jacobi':
            print('DIIS dimension:                          ', self._diis_max_dim)
        else:
            print('DIIS dimension:                           Disabled')
        print('Maximum number of iterations:            ',  self._opt_maxiter)
        print('Residual-norm threshold:                 ',  res_thrsh_str)

        print('Operator pool type:                      ',  str(self._pool_type))


    def print_summary_banner(self):

        print('\n\n                   ==> UCC-PQE summary <==')
        print('-----------------------------------------------------------')
        print('Final UCCN-PQE Energy:                      ', round(self._Egs, 10))
        if self._max_moment_rank:
            print('Moment-corrected (MP) UCCN-PQE Energy:      ', round(self._E_mmcc_mp[0], 10))
            print('Moment-corrected (EN) UCCN-PQE Energy:      ', round(self._E_mmcc_en[0], 10))
        print('Number of operators in pool:                 ', len(self._pool_obj))
        print('Final number of amplitudes in ansatz:        ', len(self._tamps))
        print('Number of classical parameters used:         ', len(self._tamps))
        print('Number of non-zero parameters used:          ', self._n_nonzero_params)
        print('Number of CNOT gates in deepest circuit:     ', self._n_cnot)
        print('Number of Pauli term measurements:           ', self._n_pauli_trm_measures)
        print('Number of residual vector evaluations:       ', self._res_vec_evals)
        print('Number of residual element evaluations*:     ', self._res_m_evals)
        print('Number of non-zero res element evaluations:  ', int(self._res_vec_evals)*self._n_nonzero_params)

        print("\n\n")
        print(self._timer)

    def fill_excited_dets(self):
        if(self._computer_type == 'fock'):
            self.fill_excited_dets_fock()
        elif(self._computer_type == 'fci'):
            self.fill_excited_dets_fci()
        else:
            raise ValueError(f"{self._computer_type} is an unrecognized computer type.") 

    def fill_excited_dets_fock(self):
        for _, sq_op in self._pool_obj:
            # 1. Identify the excitation operator
            # occ => i,j,k,...
            # vir => a,b,c,...
            # sq_op is 1.0(a^ b^ i j) - 1.0(j^ i^ b a)

            temp_idx = sq_op.terms()[0][2][-1]
            # TODO: This code assumes that the first N orbitals are occupied, and the others are virtual.
            # Use some other mechanism to identify the occupied orbitals, so we can use use PQE on excited
            # determinants.
            if temp_idx < int(sum(self._ref)/2): # if temp_idx is an occupied idx
                sq_creators = sq_op.terms()[0][1]
                sq_annihilators = sq_op.terms()[0][2]
            else:
                sq_creators = sq_op.terms()[0][2]
                sq_annihilators = sq_op.terms()[0][1]

            # 2. Get the bit representation of the sq_ex_op acting on the reference.
            # We determine the projective condition for this amplitude by zero'ing this residual.

            # `destroyed` exists solely for error catching.
            destroyed = False

            excited_det = qforte.QubitBasis(self._nqb)
            for k, occ in enumerate(self._ref):
                excited_det.set_bit(k, occ)

            # loop over annihilators
            for p in reversed(sq_annihilators):
                if( excited_det.get_bit(p) == 0):
                    destroyed=True
                    break

                excited_det.set_bit(p, 0)

            # then over creators
            for p in reversed(sq_creators):
                if (excited_det.get_bit(p) == 1):
                    destroyed=True
                    break

                excited_det.set_bit(p, 1)

            if destroyed:
                raise ValueError("no ops should destroy reference, something went wrong!!")

            I = excited_det.add()

            qc_temp = qforte.Computer(self._nqb)
            qc_temp.apply_circuit(self._Uprep)
            qc_temp.apply_operator(sq_op.jw_transform(self._qubit_excitations))
            phase_factor = qc_temp.get_coeff_vec()[I]

            self._excited_dets.append((I, phase_factor))

    def fill_excited_dets_fci(self):
        qc = qforte.FCIComputer(
            self._nel, 
            self._2_spin, 
            self._norb)
        
        for _, sq_op in self._pool_obj:
            qc.hartree_fock()
            qc.apply_sqop(sq_op)
            non_zero_tidxs = qc.get_state().get_nonzero_tidxs()

            if(len(non_zero_tidxs) != 1):
                raise ValueError("Pool object elements should only create a single excitation from hf reference.")
            
            if(len(non_zero_tidxs[0]) != 2):
                raise ValueError("Tensor indxs must be from a a matrix.")
            
            phase_factor = qc.get_state().get(non_zero_tidxs[0])

            if(phase_factor != 0.0):
                self._excited_dets_fci_comp.append((non_zero_tidxs[0], phase_factor))

    def get_residual_vector(self, trial_amps):
        if(self._computer_type == 'fock'):
            return self.get_residual_vector_fock(trial_amps)
        elif(self._computer_type == 'fci'):
            return self.get_residual_vector_fci(trial_amps)
        else:
            raise ValueError(f"{self._computer_type} is an unrecognized computer type.") 

    def get_residual_vector_fock(self, trial_amps):
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

        self._res_vec_norm = np.linalg.norm(residuals)
        self._res_vec_evals += 1
        self._res_m_evals += len(self._tamps)

        return residuals
    
    def get_residual_vector_fci(self, trial_amps):
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
        
        if not self._ref_from_hf:
            raise ValueError('get_residual_vector_fci_comp only compatible with hf reference at this time.')
        
        temp_pool = qforte.SQOpPool()

        # NICK: Write a 'updatte_coeffs' type fucntion for the op-pool.
        for tamp, top in zip(trial_amps, self._tops):
            temp_pool.add(tamp, self._pool_obj[top][1])

        qc_res = qforte.FCIComputer(
            self._nel, 
            self._2_spin, 
            self._norb)
        
        qc_res.hartree_fock()


        # function assumers first order trotter, with 1 trotter step, and time = 1.0
        qc_res.evolve_pool_trotter_basic(
            temp_pool,
            antiherm=True,
            adjoint=False)

        if(self._apply_ham_as_tensor):
            qc_res.apply_tensor_spat_012bdy(
                self._nuclear_repulsion_energy, 
                self._mo_oeis, 
                self._mo_teis, 
                self._mo_teis_einsum, 
                self._norb)
        else:   
            qc_res.apply_sqop(self._sq_ham)

        qc_res.evolve_pool_trotter_basic(
            temp_pool,
            antiherm=True,
            adjoint=True)

        R = qc_res.get_state_deep()
        residuals = []

        for IaIb, phase_factor in self._excited_dets_fci_comp:

            # Get the residual element, after accounting for numerical noise.
            res_m = R.get(IaIb) * phase_factor
            if(np.imag(res_m) != 0.0):
                raise ValueError("residual has imaginary component, something went wrong!!")

            if(self._noise_factor > 1e-12):
                res_m = np.random.normal(np.real(res_m), self._noise_factor)

            residuals.append(res_m)

        self._res_vec_norm = np.linalg.norm(residuals)
        self._res_vec_evals += 1
        self._res_m_evals += len(self._tamps)

        self._curr_energy = qc_res.get_hf_dot()

        return residuals

    def initialize_ansatz(self):
        """Adds all operators in the pool to the list of operators in the circuit,
        with amplitude 0.
        """
        for l in range(len(self._pool_obj)):
            self._tops.append(l)
            self._tamps.append(0.0)

UCCNPQE.jacobi_solver = optimizer.jacobi_solver
UCCNPQE.scipy_solver = optimizer.scipy_solver
UCCNPQE.construct_moment_space = moment_energy_corrections.construct_moment_space
UCCNPQE.compute_moment_energies = moment_energy_corrections.compute_moment_energies
