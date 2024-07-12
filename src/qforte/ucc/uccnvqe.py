"""
UCCNVQE classes
====================================
Classes for using an experiment to execute the variational quantum eigensolver
for a Trotterized (disentangeld) UCCN ansatz with fixed operators.
"""

import qforte
from qforte.abc.uccvqeabc import UCCVQE

from qforte.experiment import *
from qforte.maths import optimizer
from qforte.utils.transforms import *
from qforte.utils.state_prep import *
from qforte.utils.trotterization import trotterize
from qforte.utils import moment_energy_corrections

import numpy as np
from scipy.optimize import minimize


class UCCNVQE(UCCVQE):
    """A class that encompasses the three components of using the variational
    quantum eigensolver to optimize a parameterized disentangled UCCN-like
    wave function. (1) prepares a quantum state on the quantum computer
    representing the wave function to be simulated, (2) evauates the energy and
    gradients (3) optemizes the the wave funciton by minimizing the energy

    Attributes
    ----------
    _results : list
        The optimizer result objects from each iteration of UCCN-VQE.

    _energies : list
        The optimized energies from each iteration of UCCN-VQE.

    _grad_norms : list
        The gradient norms from each iteration of UCCN-VQE.

    """

    def run(
        self,
        opt_thresh=1.0e-5,
        opt_ftol=1.0e-5,
        opt_maxiter=200,
        pool_type="SD",
        optimizer="BFGS",
        use_analytic_grad=True,
        noise_factor=0.0,
    ):
        self._opt_thresh = opt_thresh
        self._opt_ftol = opt_ftol
        self._opt_maxiter = opt_maxiter
        self._use_analytic_grad = use_analytic_grad
        self._optimizer = optimizer
        if self._use_analytic_grad and self._optimizer in {
            "nelder-mead",
            "powell",
            "cobyla",
        }:
            print(f"{self._optimizer} optimizer doesn't support analytic grads.")
            self._use_analytic_grad = False
        self._pool_type = pool_type
        self._noise_factor = noise_factor

        self._tops = []
        self._tamps = []
        self._conmutator_pool = []
        self._converged = 0

        self._n_classical_params = 0
        self._n_cnot = 0
        self._n_pauli_trm_measures = 0
        self._res_vec_evals = 0
        self._res_m_evals = 0
        self._k_counter = 0

        self._curr_grad_norm = 0.0

        # Print options banner (should done for all algorithms).
        self.print_options_banner()

        ######### UCCN-VQE #########
        self.fill_pool()
        if self._verbose:
            print(self._pool_obj.str())

        self.initialize_ansatz()

        if self._verbose:
            print("\nt operators included from pool: \n", self._tops)
            print("\nInitial tamplitudes for tops: \n", self._tamps)

        self.solve()

        if self._max_moment_rank:
            print("\nConstructing Moller-Plesset and Epstein-Nesbet denominators")
            self.construct_moment_space()
            print("\nComputing non-iterative energy corrections")
            self.compute_moment_energies()

        if self._verbose:
            print("\nt operators included from pool: \n", self._tops)
            print("\nFinal tamplitudes for tops: \n", self._tamps)

        ######### UCCSD-VQE #########
        self._n_nonzero_params = 0
        for tmu in self._tamps:
            if np.abs(tmu) > 1.0e-12:
                self._n_nonzero_params += 1

        # verify that required attributes were defined
        # (should be called for all algorithms!)
        self.verify_run()

        self.print_summary_banner()

    # Define Algorithm abstract methods.
    def run_realistic(self):
        raise NotImplementedError(
            "run_realistic() is not fully implemented for UCCN-VQE."
        )

    def verify_run(self):
        self.verify_required_attributes()
        self.verify_required_VQE_attributes()
        self.verify_required_UCCVQE_attributes()

    def print_options_banner(self):
        print("\n-----------------------------------------------------")
        print("          Unitary Coupled Cluster VQE   ")
        print("-----------------------------------------------------")

        print("\n\n               ==> UCCN-VQE options <==")
        print("---------------------------------------------------------")
        # General algorithm options.

        self.print_generic_options()

        print("Use qubit excitations:                   ", self._qubit_excitations)
        print("Use compact excitation circuits:         ", self._compact_excitations)

        # VQE options.
        opt_thrsh_str = "{:.2e}".format(self._opt_thresh)
        print("Optimization algorithm:                  ", self._optimizer)
        print("Optimization maxiter:                    ", self._opt_maxiter)
        print("Optimizer grad-norm threshold (theta):   ", opt_thrsh_str)

        # UCCVQE options.
        print("Use analytic gradient:                   ", str(self._use_analytic_grad))
        print("Operator pool type:                      ", str(self._pool_type))

    def print_summary_banner(self):
        print("\n\n                ==> UCCN-VQE summary <==")
        print("-----------------------------------------------------------")
        print("Final UCCN-VQE Energy:                      ", round(self._Egs, 10))
        if self._max_moment_rank:
            print(
                "Moment-corrected (MP) UCCN-VQE Energy:      ",
                round(self._E_mmcc_mp[0], 10),
            )
            print(
                "Moment-corrected (EN) UCCN-VQE Energy:      ",
                round(self._E_mmcc_en[0], 10),
            )
        print("Number of operators in pool:                 ", len(self._pool_obj))
        print("Final number of amplitudes in ansatz:        ", len(self._tamps))
        print(
            "Total number of Hamiltonian measurements:    ",
            self.get_num_ham_measurements(),
        )
        print(
            "Total number of commutator measurements:     ",
            self.get_num_commut_measurements(),
        )
        print("Number of classical parameters used:         ", self._n_classical_params)
        print("Number of non-zero parameters used:          ", self._n_nonzero_params)
        print("Number of CNOT gates in deepest circuit:     ", self._n_cnot)
        print(
            "Number of Pauli term measurements:           ", self._n_pauli_trm_measures
        )

        print("Number of grad vector evaluations:           ", self._res_vec_evals)
        print("Number of individual grad evaluations:       ", self._res_m_evals)

    def solve(self):
        if self._optimizer.lower() == "jacobi":
            self.build_orb_energies()
            return self.jacobi_solver()
        else:
            return self.scipy_solve()

    def scipy_solve(self):
        # Construct arguments to hand to the minimizer.
        opts = {}

        # Options common to all minimization algorithms
        opts["disp"] = True
        opts["maxiter"] = self._opt_maxiter

        # Optimizer-specific options
        if self._optimizer in ["BFGS", "CG", "L-BFGS-B", "TNC", "trust-constr"]:
            opts["gtol"] = self._opt_thresh
        if self._optimizer == "Nelder-Mead":
            opts["fatol"] = self._opt_ftol
        if self._optimizer in ["Powell", "L-BFGS-B", "TNC", "SLSQP"]:
            opts["ftol"] = self._opt_ftol
        if self._optimizer == "COBYLA":
            opts["tol"] = self._opt_ftol
        if self._optimizer in ["L-BFGS-B", "TNC"]:
            opts["maxfun"] = self._opt_maxiter

        x0 = copy.deepcopy(self._tamps)
        init_gues_energy = self.energy_feval(x0)
        self._prev_energy = init_gues_energy

        if self._use_analytic_grad:
            print("  \n--> Begin opt with analytic gradient:")
            print(f" Initial guess energy:              {init_gues_energy:+12.10f}")

            res = minimize(
                self.energy_feval,
                x0,
                method=self._optimizer,
                jac=self.gradient_ary_feval,
                options=opts,
                callback=self.report_iteration,
            )

            # account for paulit term measurement for gradient evaluations
            # for m in range(len(self._tamps)):
            #     self._n_pauli_trm_measures += self._Nm[m] * self._Nl * res.njev

            for tmu in res.x:
                if np.abs(tmu) > 1.0e-12:
                    self._n_pauli_trm_measures += int(2 * self._Nl * res.njev)

            self._n_pauli_trm_measures += int(self._Nl * res.nfev)

        else:
            print("  \n--> Begin opt with grad estimated using first-differences:")
            print(f" Initial guess energy:              {init_gues_energy:+12.10f}")
            res = minimize(
                self.energy_feval,
                x0,
                method=self._optimizer,
                options=opts,
                callback=self.report_iteration,
            )

            # account for pauli term measurement for energy evaluations
            self._n_pauli_trm_measures += self._Nl * res.nfev

        if res.success:
            print("  => Minimization successful!")
        else:
            print("  => WARNING: minimization result may not be tightly converged.")
        print(f"  => Minimum Energy: {res.fun:+12.10f}")
        self._Egs = res.fun
        if self._optimizer == "POWELL":
            print(type(res.fun))
            print(res.fun)
            self._Egs = res.fun[()]
        self._final_result = res
        self._tamps = list(res.x)

        self._n_classical_params = len(self._tamps)
        self._n_cnot = self.build_Uvqc().get_num_cnots()

    def initialize_ansatz(self):
        """Adds all operators in the pool to the list of operators in the circuit,
        with amplitude 0.
        """
        self._tops = list(range(len(self._pool_obj)))
        self._tamps = [0.0] * len(self._pool_obj)

    # TODO: change to get_num_pt_evals
    def get_num_ham_measurements(self):
        """Returns the total number of times the energy was evaluated via
        measurement of the Hamiltonian.
        """
        try:
            self._n_ham_measurements = self._final_result.nfev
            return self._n_ham_measurements
        except AttributeError:
            # TODO: Determine the number of Hamiltonian measurements
            return "Not Yet Implemented"

    # TODO: depricate this function
    def get_num_commut_measurements(self):
        # if self._use_analytic_grad:
        #     self._n_commut_measurements = self._final_result.njev * (len(self._pool_obj))
        #     return self._n_commut_measurements
        # else:
        #     return 0
        return 0


UCCNVQE.jacobi_solver = optimizer.jacobi_solver
UCCNVQE.construct_moment_space = moment_energy_corrections.construct_moment_space
UCCNVQE.compute_moment_energies = moment_energy_corrections.compute_moment_energies
