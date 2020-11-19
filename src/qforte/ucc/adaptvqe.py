"""
adaptvqe.py
====================================
A class for using an experiment to execute the variational quantum eigensolver
for the Adaptive Derivative-Assembled Pseudo-Trotter (ADAPT) anxatz.
"""

import qforte as qf

from qforte.abc.uccvqeabc import UCCVQE
from qforte.experiment import *
from qforte.utils.transforms import *
from qforte.utils.op_pools import *
from qforte.utils.state_prep import *
from qforte.utils.trotterization import trotterize

import numpy as np
from scipy.optimize import minimize

class ADAPTVQE(UCCVQE):
    """
    A class that encompases the three componants of using the variational
    quantum eigensolver to optemize a parameterized unitary CC like wave function
    comprised of adaptivly selected operators. Growing a cirquit over many iterations
    the ADAPT-VQE: (1) prepares a quantum state on the quantum computer
    representing the wave function to be simulated, (2) evauates the energy by
    measurement, and (3) optemizes the the wave funciton by minimizing the energy.

    Attributes
    ----------
    _ref : list
        The set of 1s and 0s indicating the initial quantum state.

    _nqubits : int
        The number of qubits the calculation empolys.

    _operator : QuantumOperator
        The operator to be measured (usually the Hamiltonain), mapped to a
        qubit representation.

    _avqe_thresh : float
        The gradient norm threshold to determine when the ADAPT-VQE
        algorithm has converged.

    _opt_thresh : float
        The gradient norm threshold to determine when the classical optemizer
        algorithm has converged.

    _use_fast_measurement : bool
        Whether or not to use a faster version of the algorithm that bypasses
        measurment (unphysical for quantum computer).

    _use_analytic_grad : bool
        Whether or not to use an alaystic function for the gradient to pass to
        the optemizer. If false, the optemizer will use self-generated approximate
        gradients (if BFGS algorithm is used).

    _optimizer : string
        The type of opterizer to use for the classical portion of VQE. Suggested
        algorithms are 'BFGS' or 'Nelder-Mead' although there are many options
        (see SciPy.optemize.minimize documentation).

    _trott_num : int
        The Trotter number for the calculation
        (exact in the infinte limit).

    _results : list
        The optemizer result objects from each iteration of ADAPT-VQE.

    _energies : list
        The optemized energies from each iteration of ADAPT-VQE.

    _grad_norms : list
        The gradient norms from each iteration of ADAPT-VQE.

    _curr_grad_norm : float
        The gradient norm for the current iteration of ADAPT-VQE.

    _pool_obj : SDOpPool
        An SDOpPool object corresponding to the specefied operators of
        interest.

    _pool : list of lists with tuple and float
        The list of (optionally symmetrized) singe and double excitation
        operators to consizer. represented in the form,
        [ [(p,q), t_pq], .... , [(p,q,s,r), t_pqrs], ... ]
        where p, q, r, s are idicies of normal ordered creation or anihilation
        operators.

    _tops : list
        A list of indicies representing selected operators in the pool.

    _tamps : list
        A list of amplitudes (to be optemized) representing selected
        operators in the pool.

    _comutator_pool : list
        The QuantumOperator objects representing the comutators [H, Am] of the
        Hamiltonian (H) and each member of the operator pool (Am).

    _N_samples : int
        The number of times to measure each term in _operator
        (not yet functional).

    _converged : bool
        Whether or not the ADAPT-VQE has converged according to the gradient-norm
        threshold.

    _final_energy : float
        The final ADAPT-VQE energy value.

    _final_result : Result
        The last result object from the optemizer.

    _n_ham_measurements : int
        The total number of times the energy was evaluated via
        measurement of the Hamiltoanin

    _n_comut_measurements : int
        The total number of times the comutator was evaluated via
        measurement of [H, Am].


    Methods
    -------
    fill_pool()
        Fills the pool_ with indicies pertaining spin-complete, single and
        double excitation operators according to _nocc and _nvir.

    fill_comutator_pool()
        Fills the _comutator_pool with circuits considering the _operator to
        be measured and the _pool.

    update_ansatz()
        Adds a paramater and operator to the ADAPT-VQE circuit, and checks for
        convergence.

    build_Uprep()
        Returns a QuantumCircuit object corresponding to the state preparation
        circuit for the ADAPT-VQE ansatz on a given iteration.

    measure_gradient()
        Returns the measured energy gradient with respect to a single
        paramater Am.

    measure_energy()
        Returns the measured energy.

    energy_feval()
        Builds a state preparation circuit given a parameter list and returns
        the measured energy. Used as the function the minimizer calls.

    gradient_ary_feval()
        Computes the gradients with respect to all operators currently in the
        ADAPT-VQE ansatz. Used as the jacobian the minimizer calls.

    solve()
        Runs the optimizer to mimimize the energy. Sets certain optimizer
        parameters internally.

    conv_status()
        Sets the convergence states.

    get_num_ham_measurements()
        Returns the total number of times the energy was evaluated via
        measurement of the Hamiltoanin.

    get_num_comut_measurements()
        Returns the total number of times the comutator was evaluated via
        measurement of [H, Am].

    get_final_energy()
        Returns the final energy.

    get_final_result()
        Retruns the fianl optemization result from the optemizer. Contains
        the final amplitudes used.
    """
    def run(self,
            avqe_thresh=1.0e-2,
            pool_type='sa_SD',
            opt_thresh=1.0e-5,
            opt_maxiter=200,
            adapt_maxiter=20,
            optimizer='BFGS',
            use_analytic_grad = True,
            op_select_type='gradient',
            use_comutator_grad_selection = False,
            use_cumulative_thresh = False,
            add_equiv_ops = False):

        self._avqe_thresh = avqe_thresh
        self._opt_thresh = opt_thresh
        self._adapt_maxiter = adapt_maxiter
        self._opt_maxiter = opt_maxiter
        self._use_analytic_grad = use_analytic_grad
        self._optimizer = optimizer
        self._pool_type = pool_type
        self._op_select_type = op_select_type
        self._use_comutator_grad_selection = use_comutator_grad_selection
        self._use_cumulative_thresh = use_cumulative_thresh
        self._add_equiv_ops = add_equiv_ops

        self._results = []
        self._energies = []
        self._grad_norms = []
        self._tops = []
        self._tamps = []
        self._comutator_pool = []
        self._converged = 0

        self._n_ham_measurements = 0
        self._n_comut_measurements = 0

        self._n_classical_params = 0
        self._n_cnot = 0
        self._n_cnot_lst = []
        self._n_classical_params_lst = []
        self._n_pauli_trm_measures = 0
        self._n_pauli_trm_measures_lst = []

        # Print options banner (should done for all algorithms).
        self.print_options_banner()

        ######### ADAPT-VQE #########

        self.fill_pool()

        if self._verbose:
            print('\n\n-------------------------------------')
            print('   Second Quantized Operator Pool')
            print('-------------------------------------')
            print(self._pool_obj.str())

        # if self._op_select_type == 'minimize' and self._use_analytic_grad==False:
        #     pass
        # else:
        #     self.fill_comutator_pool()

        if self._use_comutator_grad_selection:
            self.fill_comutator_pool()

        avqe_iter = 0
        hit_maxiter = 0

        if (self._print_summary_file):
            f = open("summary.dat", "w+", buffering=1)
            f.write(f"#{'Iter(k)':>8}{'E(k)':>14}{'N(params)':>17}{'N(CNOT)':>18}{'N(measure)':>20}\n")
            f.write('#-------------------------------------------------------------------------------\n')

        while not self._converged:

            print('\n\n -----> ADAPT-VQE iteration ', avqe_iter, ' <-----\n')
            self.update_ansatz()

            if self._converged:
                break

            if(self._verbose):
                print('\ntoperators included from pool: \n', self._tops)
                print('\ntamplitudes for tops: \n', self._tamps)

            self.solve()

            if(self._verbose):
                print('\ntamplitudes for tops post solve: \n', np.real(self._tamps))

            if (self._print_summary_file):
                f.write(f'  {avqe_iter:7}    {self._energies[-1]:+15.9f}    {len(self._tamps):8}        {self._n_cnot_lst[-1]:10}        {sum(self._n_pauli_trm_measures_lst):12}\n')

            avqe_iter += 1

            if avqe_iter > self._adapt_maxiter-1:
                hit_maxiter = 1
                break

        if (self._print_summary_file):
            f.close()

        # Set final ground state energy.
        if hit_maxiter:
            self._Egs = self.get_final_energy(hit_max_avqe_iter=1)
            self._final_result = self._results[-1]

        self._Egs = self.get_final_energy()

        print('\n\n')
        print(f"{'Iter(k)':>8}{'E(k)':>14}{'N(params)':>17}{'N(CNOT)':>18}{'N(measure)':>20}")
        print('-------------------------------------------------------------------------------')

        for k, Ek in enumerate(self._energies):
            print(f' {k:7}    {Ek:+15.9f}    {self._n_classical_params_lst[k]:8}        {self._n_cnot_lst[k]:10}        {sum(self._n_pauli_trm_measures_lst[:k+1]):12}')

        self._n_classical_params = len(self._tamps)
        self._n_cnot = self._n_cnot_lst[-1]
        self._n_pauli_trm_measures = sum(self._n_pauli_trm_measures_lst)
        ######### ADAPT-VQE #########

        # Print summary banner (should done for all algorithms).
        self.print_summary_banner()

        # verify that required attributes were defined
        # (should be called for all algorithms!)
        self.verify_run()

    # Define Algorithm abstract methods.
    def run_realistic(self):
        raise NotImplementedError('run_realistic() is not fully implemented for ADAPT-VQE.')

    def verify_run(self):
        self.verify_required_attributes()
        self.verify_required_VQE_attributes()
        self.verify_required_UCCVQE_attributes()

    def print_options_banner(self):
        print('\n-----------------------------------------------------')
        print('  Adaptive Derivative-Assembled Pseudo-Trotter VQE   ')
        print('-----------------------------------------------------')

        print('\n\n               ==> ADAPT-VQE options <==')
        print('---------------------------------------------------------')
        # General algorithm options.
        print('Trial reference state:                   ',  ref_string(self._ref, self._nqb))
        print('Number of Hamiltonian Pauli terms:       ',  self._Nl)
        print('Trial state preparation method:          ',  self._trial_state_type)
        print('Trotter order (rho):                     ',  self._trotter_order)
        print('Trotter number (m):                      ',  self._trotter_number)
        print('Use fast version of algorithm:           ',  str(self._fast))
        if(self._fast):
            print('Measurement varience thresh:             ',  'NA')
        else:
            print('Measurement varience thresh:             ',  0.01)


        # VQE options.
        opt_thrsh_str = '{:.2e}'.format(self._opt_thresh)
        avqe_thrsh_str = '{:.2e}'.format(self._avqe_thresh)
        print('Optimization algorithm:                  ',  self._optimizer)
        print('Optimization maxiter:                    ',  self._opt_maxiter)
        print('Optimizer grad-norm threshold (theta):   ',  opt_thrsh_str)

        # UCCVQE options.
        print('Use analytic gradient:                   ',  str(self._use_analytic_grad))
        print('Operator pool type:                      ',  str(self._pool_type))

        # Specific ADAPT-VQE options.
        print('ADAPT-VQE operator selection type:       ',  self._op_select_type)
        print('ADAPT-VQE grad-norm threshold (eps):     ',  avqe_thrsh_str)
        print('ADAPT-VQE maxiter:                       ',  self._adapt_maxiter)


    def print_summary_banner(self):

        print('\n\n                ==> ADAPT-VQE summary <==')
        print('-----------------------------------------------------------')
        print('Final ADAPT-VQE Energy:                     ', round(self._Egs, 10))
        print('Number of operators in pool:                 ', len(self._pool))
        print('Final number of amplitudes in ansatz:        ', len(self._tamps))
        print('Total number of Hamiltonian measurements:    ', self.get_num_ham_measurements())
        print('Total number of comutator measurements:      ', self.get_num_comut_measurements())
        print('Number of classical parameters used:         ', self._n_classical_params)
        print('Number of CNOT gates in deepest circuit:     ', self._n_cnot)
        print('Number of Pauli term measurements:           ', self._n_pauli_trm_measures)

        print('Number of grad vector evaluations:           ', self._grad_vec_evals)
        print('Number of individual grad evaluations:       ', self._grad_m_evals)


    # Define VQE abstract methods.
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

        self._k_counter = 0

        opts = {}
        opts['gtol'] = self._opt_thresh
        opts['disp'] = False
        opts['maxiter'] = self._opt_maxiter
        x0 = copy.deepcopy(self._tamps)
        init_gues_energy = self.energy_feval(x0)

        if self._use_analytic_grad:
            print('  \n--> Begin opt with analytic graditent:')
            print(f" Initail guess energy:              {init_gues_energy:+12.10f}")

            res =  minimize(self.energy_feval, x0,
                                    method=self._optimizer,
                                    jac=self.gradient_ary_feval,
                                    options=opts,
                                    callback=self.callback)

            # account for energy evaluations
            self._n_pauli_measures_k += self._Nl * res.nfev

            # account for gradient evaluations
            for m in self._tops:
                self._n_pauli_measures_k += self._Nm[m] * self._Nl * res.njev

        else:
            print('  \n--> Begin opt with grad estimated using first-differences:')
            print(f" Initail guess energy:              {init_gues_energy:+12.10f}")
            res =  minimize(self.energy_feval, x0,
                                    method=self._optimizer,
                                    options=opts,
                                    callback=self.callback)

            self._n_pauli_measures_k += self._Nl * res.nfev

        if(res.success):
            print('  => Minimization successful!')
            print(f'  => Minimum Energy: {res.fun:+12.10f}')

        else:
            print('  => WARNING: minimization result may not be tightly converged.')
            print(f'  => Minimum Energy: {res.fun:+12.10f}')

        self._energies.append(res.fun)
        self._results.append(res)
        self._tamps = list(res.x)
        self._n_pauli_trm_measures_lst.append(self._n_pauli_measures_k)
        self._n_cnot_lst.append(self.build_Uvqc().get_num_cnots())

    # Define ADAPT-VQE methods.
    def update_ansatz(self):
        self._n_pauli_measures_k = 0
        if (self._op_select_type=='gradient'):
            curr_norm = 0.0
            lgrst_grad = 0.0
            Uvqc = self.build_Uvqc()

            if self._verbose:
                print('     op index (m)     N pauli terms              Gradient            Tmu  ')
                print('  ------------------------------------------------------------------------------')

            if self._use_comutator_grad_selection:
                grads = self.measure_comutator_gradient(self._comutator_pool, Uvqc)
            else:
                # grads = self.measure_gradient(use_entire_pool=True)
                grads = self.measure_gradient3()

            for m, grad_m in enumerate(grads):
                if self._use_comutator_grad_selection:
                    self._n_pauli_measures_k += len(self._comutator_pool.terms()[m][1].terms())
                else:
                    # referes to number of times sigma_y must be measured in "stratagies for UCC" grad eval circuit
                    self._n_pauli_measures_k += self._Nl * self._Nm[m]

                curr_norm += grad_m*grad_m
                if (self._verbose):
                    print(f'       {m:3}                {self._Nm[m]:8}             {grad_m:+12.9f}      {self._pool[m][1].terms()[0][1]}')

                if (abs(grad_m) > abs(lgrst_grad)):

                    if(abs(lgrst_grad) > 0.0):
                        secnd_lgst_grad = lgrst_grad
                        secnd_lgrst_grad_idx = lgrst_grad_idx

                    lgrst_grad = grad_m
                    lgrst_grad_idx = m

            curr_norm = np.sqrt(curr_norm)
            if self._use_comutator_grad_selection:
                print("\n==> Measring gradients from pool:")
                print(" Norm of <[H,Am]> = %12.8f" %curr_norm)
                print(" Max  of <[H,Am]> = %12.8f" %lgrst_grad)
            else:
                print("\n==> Measring gradients:")
                print(" Norm of g_vec = %12.8f" %curr_norm)
                print(" Max  of g_vec = %12.8f" %lgrst_grad)

            self._curr_grad_norm = curr_norm
            self._grad_norms.append(curr_norm)

            self.conv_status()

            if not self._converged:
                if(self._use_cumulative_thresh):
                    temp_order_tops = []
                    grads_sq = [(grads[m] * grads[m], m) for m in range(len(grads))]
                    grads_sq.sort()
                    gm_sq_sum = 0.0
                    for m, gm_sq in enumerate(grads_sq):
                        gm_sq_sum += gm_sq[0]
                        if gm_sq_sum > (self._avqe_thresh * self._avqe_thresh):
                            print(f"  Adding operator m =     {gm_sq[1]:10}   |gm| = {np.sqrt(gm_sq[0]):10.8f}")
                            self._tamps.append(0.0)
                            temp_order_tops.insert(0,gm_sq[1])

                    self._tops.extend(copy.deepcopy(temp_order_tops))
                    self._n_classical_params_lst.append(len(self._tops))
                else:
                    print("  Adding operator m =", lgrst_grad_idx)
                    self._tops.append(lgrst_grad_idx)
                    self._tamps.append(0.0)

                    if(self._add_equiv_ops):
                        if (abs(lgrst_grad) - abs(secnd_lgst_grad) < 1.0e-5):
                            print(" *Adding operator m =", secnd_lgrst_grad_idx)
                            self._tops.append(secnd_lgrst_grad_idx)
                            self._tamps.append(0.0)

                    self._n_classical_params_lst.append(len(self._tops))

            else:
                print("\n  ADAPT-VQE converged!")

        elif(self._op_select_type=='minimize'):
        # TODO(Nick): remove or fix this option to work correctly

            if not self._use_comutator_grad_selection:
                raise ValueError("must use computator gradients for 'minimization' selection type")

            print("==> Minimizing candidate amplitude from pool:")
            opts = {}
            opts['gtol'] = self._opt_thresh
            opts['disp'] = False
            opts['maxiter'] = self._opt_maxiter

            x0 = [0.0]
            self._trial_op = 0
            init_gues_energy = self.energy_feval2(x0)

            if self._use_analytic_grad:
                print('  \n--> Begin selection opt with analytic graditent:')
                print('  Initial guess energy: ', round(init_gues_energy,10))

            else:
                print('  \n--> Begin selection opt with grad estimated using first-differences:')
                print('  Initial guess energy: ', round(init_gues_energy,10))

            if self._verbose:
                print('\n')
                print('     op index (m)          Energy decrease')
                print('  -----------------------------------------------')

            for m in range(len(self._pool)):
                self._trial_op = m
                if self._use_analytic_grad:
                    res =  minimize(self.energy_feval2, x0,
                                    method=self._optimizer,
                                    jac=self.gradient_ary_feval2,
                                    options=opts)

                    if self._verbose:
                        print('       ', m, '                     ', '{:+.09f}'.format(res.fun-init_gues_energy))

                else:
                    res =  minimize(self.energy_feval2, x0,
                                    method=self._optimizer,
                                    options=opts)

                    if self._verbose:
                        print('       ', m, '                     ', '{:+.09f}'.format(res.fun-init_gues_energy))


                self._n_ham_measurements += res.nfev
                self._n_pauli_measures_k += self._Nl * res.nfev

                if (self._use_analytic_grad):
                    self._n_comut_measurements += res.njev
                    self._n_pauli_measures_k += len(self._comutator_pool.terms()[m][1].terms()) * res.njev

                if(m==0):
                    min_amp_e = res.fun
                    min_amp_idx = m

                else:
                    if(res.fun < min_amp_e):
                        min_amp_e = res.fun
                        min_amp_idx = m

            print("  Adding operator m =", min_amp_idx)
            self._tops.append(min_amp_idx)
            self._tamps.append(res.x[0])

        else:
            raise ValueError('Invalid value specified for _op_select_type')

    def build_Uvqc2(self, param):
        """ This function returns the QuantumCircuit object built
        from the appropiate ampltudes (tops)

        Parameters
        ----------
        param : float
            A single parameter to opteimze appended to current _tamps.
        """
        sq_ops = []
        new_tops  = copy.deepcopy(self._tops)
        new_tamps = copy.deepcopy(self._tamps)
        new_tops.append(self._trial_op)
        new_tamps.append(param)

        temp_pool = qf.SQOpPool()
        for tamp, top in zip(new_tamps, new_tops):
            temp_pool.add_term(tamp, self._pool[top][1])

        A = temp_pool.get_quantum_operator('comuting_grp_lex')

        U, phase1 = trotterize(A, trotter_number=self._trotter_number)
        Uvqc = qf.QuantumCircuit()
        Uvqc.add_circuit(self._Uprep)
        Uvqc.add_circuit(U)
        if phase1 != 1.0 + 0.0j:
            raise ValueError("Encountered phase change, phase not equal to (1.0 + 0.0i)")

        return Uvqc

    def energy_feval2(self, params):
        Ucirc = self.build_Uvqc2(params[0])
        return self.measure_energy(Ucirc)

    # this fuction needs to change to use updated gradient measurement
    def gradient_ary_feval2(self, param):
        Uvqc = self.build_Uvqc2(params[0])
        grads = self.measure_comutator_gradient(self._comutator_pool, Uvqc, [self._trial_op])
        return grads

    def conv_status(self):
        if abs(self._curr_grad_norm) < abs(self._avqe_thresh):
            self._converged = 1
            self._final_energy = self._energies[-1]
            self._final_result = self._results[-1]
        else:
            self._converged = 0

    def get_num_ham_measurements(self):
        for res in self._results:
            self._n_ham_measurements += res.nfev
        return self._n_ham_measurements

    def get_num_comut_measurements(self):
        if(self._op_select_type=='gradient'):
            self._n_comut_measurements += len(self._tamps) * len(self._pool)

        if self._use_analytic_grad:
            for m, res in enumerate(self._results):
                self._n_comut_measurements += res.njev * (m+1)

        return self._n_comut_measurements

    def get_final_energy(self, hit_max_avqe_iter=0):
        """
        Parameters
        ----------
        hit_max_avqe_iter : bool
            Wether or not to use the ADAPT-VQE has already hit the maximum
            number of iterations.
        """
        if hit_max_avqe_iter:
            print("\nADAPT-VQE at maximum number of iterations!")
            self._final_energy = self._energies[-1]
        else:
            return self._final_energy

    def get_final_result(self, hit_max_avqe_iter=0):
        """
        Parameters
        ----------
        hit_max_avqe_iter : bool
            Wether or not to use the ADAPT-VQE has already hit the maximum
            number of iterations.
        """
        if hit_max_avqe_iter:
            self._final_result = self._results[-1]
        else:
            return self._final_result
