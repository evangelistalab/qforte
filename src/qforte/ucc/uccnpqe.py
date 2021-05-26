"""
uccpqe.py
====================================
A class for solving the schrodinger equation via measurement of its projections
and subsequent updates of the UCC amplitudes.
"""

import qforte
from qforte.abc.uccpqeabc import UCCPQE

from qforte.experiment import *
from qforte.utils.transforms import *
from qforte.utils.op_pools import *
from qforte.utils.state_prep import *
from qforte.utils.trotterization import trotterize

from qforte.helper.printing import matprint

import numpy as np
from scipy.linalg import lstsq

class UCCNPQE(UCCPQE):
    """
    A class that encompasses the three components of using the variational
    quantum eigensolver to optimize a parameterized unitary CCSD like wave function.

    UCC-PQE: (1) prepares a quantum state on the quantum computer
    representing the wave function to be simulated, (2) evaluates the residuals by
    measurement, and (3) optimizes the wave fuction via projective solution of
    the UCC Schrodinger Equation.

    Attributes
    ----------
    _ref : list
        The set of 1s and 0s indicating the initial quantum state.

    """
    def run(self,
            pool_type='SD',
            res_vec_thresh = 1.0e-5,
            diis_maxiter = 40,
            noise_factor = 0.0):

        if(self._state_prep_type != 'occupation_list'):
            raise ValueError("PQE implementation can only handle occupation_list Hartree-Fock reference.")

        self._pool_type = pool_type
        self._res_vec_thresh = res_vec_thresh
        self._diis_maxiter = diis_maxiter
        self._noise_factor = noise_factor

        self._tops = []
        self._tamps = []
        self._converged = 0

        self._res_vec_evals = 0
        self._res_m_evals = 0

        self._n_classical_params = 0
        self._n_cnot = 0
        self._n_pauli_trm_measures = 0
        # self._results = [] #keep for future implementations

        self.print_options_banner()
        self.fill_pool()

        if self._verbose:
            print('\n\n-------------------------------------')
            print('   Second Quantized Operator Pool')
            print('-------------------------------------')
            print(self._pool_obj.str())

        self.initialize_ansatz()

        if(self._verbose):
            print('\nt operators included from pool: \n', self._tops)
            print('Initial tamplitudes for tops: \n', self._tamps)

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

        res_thrsh_str = '{:.2e}'.format(self._res_vec_thresh)
        print('DIIS maxiter:                            ',  self._diis_maxiter) # RENAME
        print('DIIS res-norm threshold:                 ',  res_thrsh_str)

        print('Operator pool type:                      ',  str(self._pool_type))


    def print_summary_banner(self):

        print('\n\n                   ==> UCC-PQE summary <==')
        print('-----------------------------------------------------------')
        print('Final UCCN-PQE Energy:                      ', round(self._Egs, 10))
        print('Number of operators in pool:                 ', len(self._pool))
        print('Final number of amplitudes in ansatz:        ', len(self._tamps))
        print('Number of classical parameters used:         ', len(self._tamps))
        print('Number of non-zero parameters used:          ', self._n_nonzero_params)
        print('Number of CNOT gates in deepest circuit:     ', self._n_cnot)
        print('Number of Pauli term measurements:           ', self._n_pauli_trm_measures)
        print('Number of residual vector evaluations:       ', self._res_vec_evals)
        print('Number of residual element evaluations*:     ', self._res_m_evals)
        print('Number of non-zero res element evaluations:  ', int(self._res_vec_evals)*self._n_nonzero_params)

    def solve(self):
        """
        Parameters
        ----------
        fast : bool
            Wether or not to use the optemized but unphysical energy evaluation
            function.
        maxiter : int
            The maximum number of iterations for the scipy optimizer.
        """

        self.diis_solve()

    def diis_solve(self):
        # draws heavy insiration from Daniel Smith's ccsd_diss.py code in psi4 numpy
        diis_dim = 0
        t_diis = [copy.deepcopy(self._tamps)]
        e_diis = []
        rk_norm = 1.0
        Ek0 = self.energy_feval(self._tamps)

        print('\n    k iteration         Energy               dE           Nrvec ev      Nrm ev*        ||r||')
        print('---------------------------------------------------------------------------------------------------')

        if (self._print_summary_file):
            f = open("summary.dat", "w+", buffering=1)
            f.write('\n#    k iteration         Energy               dE           Nrvec ev      Nrm ev*        ||r||')
            f.write('\n#--------------------------------------------------------------------------------------------------')
            f.close()

        for k in range(1, self._diis_maxiter+1):

            t_old = copy.deepcopy(self._tamps)

            #do regular update
            r_k = self.get_residual_vector(self._tamps)
            rk_norm = np.linalg.norm(r_k)

            r_k = self.get_res_over_mpdenom(r_k)
            self._tamps = list(np.add(self._tamps, r_k))

            Ek = self.energy_feval(self._tamps)
            dE = Ek - Ek0
            Ek0 = Ek

            self._res_vec_evals += 1
            self._res_m_evals += len(self._tamps)

            print(f'     {k:7}        {Ek:+12.10f}      {dE:+12.10f}      {self._res_vec_evals:4}        {self._res_m_evals:6}       {rk_norm:+12.10f}')

            if (self._print_summary_file):
                f = open("summary.dat", "a", buffering=1)
                f.write(f'\n     {k:7}        {Ek:+12.12f}      {dE:+12.12f}      {self._res_vec_evals:4}        {self._res_m_evals:6}       {rk_norm:+12.12f}')
                f.close()

            if(rk_norm < self._res_vec_thresh):
                self._Egs = Ek
                break

            t_diis.append(copy.deepcopy(self._tamps))
            e_diis.append(np.subtract(copy.deepcopy(self._tamps), t_old))

            if(k >= 1):
                diis_dim = len(t_diis) - 1

                # Construct diis B matrix (following Crawford Group github tutorial)
                B = np.ones((diis_dim+1, diis_dim+1)) * -1
                bsol = np.zeros(diis_dim+1)

                B[-1, -1] = 0.0
                bsol[-1] = -1.0
                for i, ei in enumerate(e_diis):
                    for j, ej in enumerate(e_diis):
                        B[i,j] = np.dot(np.real(ei), np.real(ej))

                B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()

                x = np.linalg.solve(B, bsol)

                t_new = np.zeros(( len(self._tamps) ))
                for l in range(diis_dim):
                    temp_ary = x[l] * np.asarray(t_diis[l+1])
                    t_new = np.add(t_new, temp_ary)

                self._tamps = copy.deepcopy(t_new)

        self._n_classical_params = self._n_classical_params = len(self._tamps)
        self._n_cnot = self.build_Uvqc().get_num_cnots()
        self._n_pauli_trm_measures += 2*self._Nl*k*len(self._tamps) + self._Nl*k
        self._Egs = Ek

    def get_residual_vector(self, trial_amps):
        if(self._pool_type == 'sa_SD'):
            raise ValueError('Must use single term particle-hole nbody operators for residual calculation')

        temp_pool = qforte.SQOpPool()
        for param, top in zip(trial_amps, self._tops):
            temp_pool.add(param, self._pool[top][1])

        A = temp_pool.get_quantum_operator('commuting_grp_lex')
        U, U_phase = trotterize(A, trotter_number=self._trotter_number)
        if U_phase != 1.0 + 0.0j:
            raise ValueError("Encountered phase change, phase not equal to (1.0 + 0.0i)")

        qc_res = qforte.Computer(self._nqb)
        qc_res.apply_circuit(self._Uprep)
        qc_res.apply_circuit(U)
        qc_res.apply_operator(self._qb_ham)
        qc_res.apply_circuit(U.adjoint())

        coeffs = qc_res.get_coeff_vec()
        residuals = []

        for m in self._tops:
            # 1. Identify the excitation operator
            sq_op = self._pool[m][1]
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

            excited_det = qforte.QuantumBasis(self._nqb)
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

            # 3. Compute the phase of the operator, relative to its determinant.
            qc_temp = qforte.Computer(self._nqb)
            qc_temp.apply_circuit(self._Uprep)
            qc_temp.apply_operator(sq_op.jw_transform())
            phase_factor = qc_temp.get_coeff_vec()[I]

            # 4. Get the residual element, after accounting for numerical noise.
            res_m = coeffs[I] * phase_factor
            if(np.imag(res_m) != 0.0):
                raise ValueError("residual has imaginary component, something went wrong!!")

            if(self._noise_factor > 1e-12):
                res_m = np.random.normal(np.real(res_m), self._noise_factor)

            residuals.append(res_m)

        return residuals

    def get_res_over_mpdenom(self, residuals):

        resids_over_denoms = []
        # each operator needs a score, so loop over toperators
        for m in self._tops:
            sq_op = self._pool[m][1]

            temp_idx = sq_op.terms()[0][2][-1]
            if temp_idx < int(sum(self._ref)/2): # if temp_idx is an occupied idx
                sq_creators = sq_op.terms()[0][1]
                sq_annihilators = sq_op.terms()[0][2]
            else:
                sq_creators = sq_op.terms()[0][2]
                sq_annihilators = sq_op.terms()[0][1]

            denom = sum(self._orb_e[x] for x in sq_annihilators) - sum(self._orb_e[x] for x in sq_creators)

            res_m = copy.deepcopy(residuals[m])
            res_m /= denom # divide by energy denominator

            resids_over_denoms.append(res_m)

        return resids_over_denoms

    def build_orb_energies(self):
        self._orb_e = []

        print('\nBuilding single particle energies list:')
        print('---------------------------------------')
        qc = qforte.Computer(self._nqb)
        qc.apply_circuit(build_Uprep(self._ref, 'occupation_list'))
        E0 = qc.direct_op_exp_val(self._qb_ham)

        for i in range(self._nqb):
            qc = qforte.Computer(self._nqb)
            qc.apply_circuit(build_Uprep(self._ref, 'occupation_list'))
            qc.apply_gate(qforte.gate('X', i, i))
            Ei = qc.direct_op_exp_val(self._qb_ham)

            if(i<sum(self._ref)):
                ei = E0 - Ei
            else:
                ei = Ei - E0

            print(f'  {i:3}     {ei:+16.12f}')
            self._orb_e.append(ei)

    def initialize_ansatz(self):
        for l in range(len(self._pool)):
            self._tops.append(l)
            self._tamps.append(0.0)
