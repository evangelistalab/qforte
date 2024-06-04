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
        r_\\mu = \\langle \\Phi_\\mu | \\hat{U}^\\dagger(\\mathbf{t}) \\hat{H} \\hat{U}(\\mathbf{t}) | \\Phi_0 \\rangle

    and (3) optimizes the wave fuction via projective solution of
    the UCC Schrodinger Equation via a quazi-Newton update equation.
    Using this strategy, an amplitude :math:`t_\\mu^{(k+1)}` for iteration :math:`k+1`
    is given by

    .. math::
        t_\\mu^{(k+1)} = t_\\mu^{(k)} + \\frac{r_\\mu^{(k)}}{\\Delta_\\mu}

    where :math:`\\Delta_\\mu` is the standard Moller Plesset denominator.

    Attributes
    ----------
    _ref : list
        The set of 1s and 0s indicating the initial quantum state.

    """

    def run(
        self,
        pool_type="SD",
        opt_thresh=1.0e-5,
        opt_maxiter=40,
        noise_factor=0.0,
        optimizer="jacobi",
    ):
        if self._state_prep_type != "occupation_list":
            raise ValueError(
                "PQE implementation can only handle occupation_list Hartree-Fock reference."
            )

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

        self._n_classical_params = 0
        self._n_cnot = 0
        self._n_pauli_trm_measures = 0
        # self._results = [] #keep for future implementations

        self.print_options_banner()
        self.fill_pool()

        if self._verbose:
            print("\n\n-------------------------------------")
            print("   Second Quantized Operator Pool")
            print("-------------------------------------")
            print(self._pool_obj.str())

        self.initialize_ansatz()

        if self._verbose:
            print("\nt operators included from pool: \n", self._tops)
            print("Initial tamplitudes for tops: \n", self._tamps)

        self.fill_excited_dets()
        self.build_orb_energies()
        self.solve()

        if self._max_moment_rank:
            print("\nConstructing Moller-Plesset and Epstein-Nesbet denominators")
            self.construct_moment_space()
            print("\nComputing non-iterative energy corrections")
            self.compute_moment_energies()

        if self._verbose:
            print("\nt operators included from pool: \n", self._tops)

            print("Final tamplitudes for tops:")
            print("------------------------------")
            for i, tamp in enumerate(self._tamps):
                print(f"  {i:4}      {tamp:+12.8f}")

        self._n_nonzero_params = 0
        for tmu in self._tamps:
            if np.abs(tmu) > 1.0e-12:
                self._n_nonzero_params += 1

        self._n_pauli_trm_measures = int(
            2 * self._Nl * self._res_vec_evals * self._n_nonzero_params
            + self._Nl * self._res_vec_evals
        )

        self.print_summary_banner()
        self.verify_run()

    def run_realistic(self):
        raise NotImplementedError(
            "run_realistic() is not fully implemented for UCCN-PQE."
        )

    def verify_run(self):
        self.verify_required_attributes()
        self.verify_required_PQE_attributes()
        self.verify_required_UCCPQE_attributes()

    def print_options_banner(self):
        print("\n-----------------------------------------------------")
        print("           Unitary Coupled Cluster PQE   ")
        print("-----------------------------------------------------")

        print("\n\n                 ==> UCC-PQE options <==")
        print("---------------------------------------------------------")

        self.print_generic_options()

        print("Use qubit excitations:                   ", self._qubit_excitations)
        print("Use compact excitation circuits:         ", self._compact_excitations)

        res_thrsh_str = "{:.2e}".format(self._opt_thresh)
        print("Optimizer:                               ", self._optimizer)
        if self._diis_max_dim >= 2 and self._optimizer.lower() == "jacobi":
            print("DIIS dimension:                          ", self._diis_max_dim)
        else:
            print("DIIS dimension:                           Disabled")
        print("Maximum number of iterations:            ", self._opt_maxiter)
        print("Residual-norm threshold:                 ", res_thrsh_str)

        print("Operator pool type:                      ", str(self._pool_type))

    def print_summary_banner(self):
        print("\n\n                   ==> UCC-PQE summary <==")
        print("-----------------------------------------------------------")
        print("Final UCCN-PQE Energy:                      ", round(self._Egs, 10))
        if self._max_moment_rank:
            print(
                "Moment-corrected (MP) UCCN-PQE Energy:      ",
                round(self._E_mmcc_mp[0], 10),
            )
            print(
                "Moment-corrected (EN) UCCN-PQE Energy:      ",
                round(self._E_mmcc_en[0], 10),
            )
        print("Number of operators in pool:                 ", len(self._pool_obj))
        print("Final number of amplitudes in ansatz:        ", len(self._tamps))
        print("Number of classical parameters used:         ", len(self._tamps))
        print("Number of non-zero parameters used:          ", self._n_nonzero_params)
        print("Number of CNOT gates in deepest circuit:     ", self._n_cnot)
        print(
            "Number of Pauli term measurements:           ", self._n_pauli_trm_measures
        )
        print("Number of residual vector evaluations:       ", self._res_vec_evals)
        print("Number of residual element evaluations*:     ", self._res_m_evals)
        print(
            "Number of non-zero res element evaluations:  ",
            int(self._res_vec_evals) * self._n_nonzero_params,
        )

    def fill_excited_dets(self):
        """Populate self._excited_dets."""
        for _, sq_op in self._pool_obj:
            # 1. Identify the excitation operator
            # occ => i,j,k,...
            # vir => a,b,c,...
            # sq_op is 1.0(a^ b^ i j) - 1.0(j^ i^ b a)

            temp_idx = sq_op.terms()[0][2][-1]
            if self._ref[temp_idx]:  # if temp_idx is an occupied idx
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
                if excited_det.get_bit(p) == 0:
                    destroyed = True
                    break

                excited_det.set_bit(p, 0)

            # then over creators
            for p in reversed(sq_creators):
                if excited_det.get_bit(p) == 1:
                    destroyed = True
                    break

                excited_det.set_bit(p, 1)

            if destroyed:
                raise ValueError(
                    "no ops should destroy reference, something went wrong!!"
                )

            I = excited_det.index()

            qc_temp = qforte.Computer(self._nqb)
            qc_temp.apply_circuit(self._refprep)
            qc_temp.apply_operator(sq_op.jw_transform(self._qubit_excitations))
            phase_factor = qc_temp.get_coeff_vec()[I]

            self._excited_dets.append((I, phase_factor))

    def get_residual_vector(self, trial_amps):
        """Returns the residual vector with elements pertaining to all operators
        in the ansatz circuit.

        Parameters
        ----------
        trial_amps : list of floats
            The list of (real) floating point numbers which will characterize
            the state preparation circuit used in calculation of the residuals.
        """
        if self._pool_type == "sa_SD":
            raise ValueError(
                "Must use single term particle-hole nbody operators for residual calculation"
            )

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
            if np.imag(res_m) != 0.0:
                raise ValueError(
                    "residual has imaginary component, something went wrong!!"
                )

            if self._noise_factor > 1e-12:
                res_m = np.random.normal(np.real(res_m), self._noise_factor)

            residuals.append(res_m)

        self._res_vec_norm = np.linalg.norm(residuals)
        self._res_vec_evals += 1
        self._res_m_evals += len(self._tamps)

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
