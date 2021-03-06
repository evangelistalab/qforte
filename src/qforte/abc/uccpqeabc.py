import qforte as qf
from abc import abstractmethod
# from qforte.abc.vqeabc import VQE
from qforte.abc.pqeabc import PQE
from qforte.utils.op_pools import *

from qforte.experiment import *
from qforte.utils.transforms import *
from qforte.utils.state_prep import ref_to_basis_idx
from qforte.utils.trotterization import trotterize

import numpy as np

class UCCPQE(PQE):

    def fill_pool(self):

        self._pool_obj = qf.SQOpPool()
        self._pool_obj.set_orb_spaces(self._ref)

        if self._pool_type in {'sa_SD', 'GSD', 'SD', 'SDT', 'SDTQ', 'SDTQP', 'SDTQPH'}:
            self._pool_obj.fill_pool(self._pool_type)
        else:
            raise ValueError('Invalid operator pool type specified.')

        self._pool = self._pool_obj.terms()

        self._grad_vec_evals = 0
        self._grad_m_evals = 0
        self._k_counter = 0
        self._grad_m_evals = 0
        self._prev_energy = self._hf_energy
        self._curr_energy = 0.0
        self._curr_grad_norm = 0.0

        self._Nm = []
        for m in range(len(self._pool_obj.terms())):
            self._Nm.append(len(self._pool_obj.terms()[m][1].jw_transform().terms()))

    # TODO (opt major): write a C function that prepares this super efficiently
    def build_Uvqc(self, params=None):
        """ This function returns the QuantumCircuit object built
        from the appropiate ampltudes (tops)

        Parameters
        ----------
        params : list
            A list of parameters define the variational degress of freedom in
            the state preparation circuit Uvqc.
        """
        temp_pool = qf.SQOpPool()
        if params is None:
            for tamp, top in zip(self._tamps, self._tops):
                temp_pool.add_term(tamp, self._pool[top][1])
        else:
            for param, top in zip(params, self._tops):
                temp_pool.add_term(param, self._pool[top][1])

        A = temp_pool.get_quantum_operator('commuting_grp_lex')

        U, phase1 = trotterize(A, trotter_number=self._trotter_number)
        Uvqc = qforte.QuantumCircuit()
        Uvqc.add_circuit(self._Uprep)
        Uvqc.add_circuit(U)
        if phase1 != 1.0 + 0.0j:
            raise ValueError("Encountered phase change, phase not equal to (1.0 + 0.0i)")

        return Uvqc

    # def measure_gradient2(self, params=None):
    #     """
    #     Parameters
    #     ----------
    #     HAm : QuantumOpPool
    #         The commutator to measure.
    #
    #     Ucirc : QuantumCircuit
    #         The state preparation circuit.
    #     """
    #
    #     if self._fast==False:
    #         raise ValueError("self._fast must be True for gradient measurement.")
    #
    #     grads = []
    #     M = len(self._tamps)
    #     omega_o = ref_to_basis_idx(self._ref)
    #
    #     for mu in range(M):
    #         lo_pool = qf.SQOpPool()
    #         hi_pool = qf.SQOpPool()
    #         if params is None:
    #             for nu in range(mu):
    #                 # nu indexes term THAT ARE ALREDY INCLUDED from the pool
    #                 tamp = self._tamps[nu] # actual amp t_nu
    #                 top = self._tops[nu] # actual pool idx for K_nu
    #                 lo_pool.add_term(tamp, self._pool[top][1])
    #
    #             for nu in range(mu, M):
    #                 tamp = self._tamps[nu]
    #                 top = self._tops[nu]
    #                 hi_pool.add_term(tamp, self._pool[top][1])
    #
    #
    #         else:
    #             for nu in range(mu):
    #                 tamp = params[nu]
    #                 top = self._tops[nu]
    #                 lo_pool.add_term(tamp, self._pool[top][1])
    #
    #             for nu in range(mu, M):
    #                 tamp = params[nu]
    #                 top = self._tops[nu]
    #                 hi_pool.add_term(tamp, self._pool[top][1])
    #
    #         Kmu = self._pool[self._tops[mu]][1].jw_transform()
    #         Kmu.mult_coeffs(self._pool[self._tops[mu]][0])
    #
    #         Alo = lo_pool.get_quantum_operator('commuting_grp_lex')
    #         Ahi = hi_pool.get_quantum_operator('commuting_grp_lex')
    #
    #         Ulo, plo = trotterize(Alo, trotter_number=self._trotter_number)
    #         Uhi, phi = trotterize(Ahi, trotter_number=self._trotter_number)
    #         if (plo != 1.0 + 0.0j) or (phi != 1.0 + 0.0j):
    #             raise ValueError("Encountered phase change, phase not equal to (1.0 + 0.0i)")
    #
    #         qc = qforte.QuantumComputer(self._nqb)
    #         qc.apply_circuit(self._Uprep)
    #         qc.apply_circuit(Ulo)
    #         qc.apply_operator(Kmu)
    #         qc.apply_circuit(Uhi)
    #         qc.apply_operator(self._qb_ham)
    #         qc.apply_circuit(Uhi.adjoint())
    #         qc.apply_circuit(Ulo.adjoint())
    #
    #         grads.append( 2.0*qc.get_coeff_vec()[omega_o] )
    #
    #
    #     for val in grads:
    #         assert(np.isclose(np.imag(val), 0.0))
    #
    #     # self._curr_grad_norm = np.linalg.norm(grads)
    #
    #     return np.real(grads)

    def measure_gradient(self, params=None, use_entire_pool=False):
        """
        Parameters
        ----------
        HAm : QuantumOpPool
            The commutator to measure.

        Ucirc : QuantumCircuit
            The state preparation circuit.
        """

        if self._fast==False:
            raise ValueError("self._fast must be True for gradient measurement.")

        grads = np.zeros(len(self._tamps))
        M = len(self._tamps)

        if use_entire_pool:
            M = len(self._pool)
            pool_amps = np.zeros(M)
            for tamp, top in zip(self._tamps, self._tops):
                pool_amps[top] = tamp
        else:
            M = len(self._tamps)

        grads = np.zeros(M)

        if params is None:
            Utot = self.build_Uvqc()
        else:
            Utot = self.build_Uvqc(params)

        qc_psi = qforte.QuantumComputer(self._nqb) # build | sig_N > according ADAPT-VQE analytial grad section
        qc_psi.apply_circuit(Utot)
        qc_sig = qforte.QuantumComputer(self._nqb) # build | psi_N > according ADAPT-VQE analytial grad section
        psi_i = copy.deepcopy(qc_psi.get_coeff_vec())
        qc_sig.set_coeff_vec(copy.deepcopy(psi_i)) # not sure if copy is faster or reapplication of state
        qc_sig.apply_operator(self._qb_ham)

        mu = M-1

        # find <sing_N | K_N | psi_N>
        if use_entire_pool:
            Kmu_prev = self._pool[mu][1].jw_transform()
            Kmu_prev.mult_coeffs(self._pool[mu][0])
        else:
            Kmu_prev = self._pool[self._tops[mu]][1].jw_transform()
            Kmu_prev.mult_coeffs(self._pool[self._tops[mu]][0])

        qc_psi.apply_operator(Kmu_prev)
        grads[mu] = 2.0 * np.real(np.vdot(qc_sig.get_coeff_vec(), qc_psi.get_coeff_vec()))

        #reset Kmu_prev |psi_i> -> |psi_i>
        qc_psi.set_coeff_vec(copy.deepcopy(psi_i))

        for mu in reversed(range(M-1)):

            # mu => N-1 => M-2
            # mu+1 => N => M-1
            # Kmu => KN-1
            # Kmu_prev => KN

            if use_entire_pool:
                tamp = pool_amps[mu+1]
            elif params is None:
                tamp = self._tamps[mu+1]
            else:
                tamp = params[mu+1]

            if use_entire_pool:
                Kmu = self._pool[mu][1].jw_transform()
                Kmu.mult_coeffs(self._pool[mu][0])
            else:
                Kmu = self._pool[self._tops[mu]][1].jw_transform()
                Kmu.mult_coeffs(self._pool[self._tops[mu]][0])

            Umu, pmu = trotterize(Kmu_prev, factor=-tamp, trotter_number=self._trotter_number)

            if (pmu != 1.0 + 0.0j):
                raise ValueError("Encountered phase change, phase not equal to (1.0 + 0.0i)")

            qc_sig.apply_circuit(Umu)
            qc_psi.apply_circuit(Umu)
            psi_i = copy.deepcopy(qc_psi.get_coeff_vec())

            qc_psi.apply_operator(Kmu)
            grads[mu] = 2.0 * np.real(np.vdot(qc_sig.get_coeff_vec(), qc_psi.get_coeff_vec()))

            #reset Kmu |psi_i> -> |psi_i>
            qc_psi.set_coeff_vec(copy.deepcopy(psi_i))
            Kmu_prev = Kmu

        # self._curr_grad_norm = np.linalg.norm(grads)

        for val in grads:
            assert(np.isclose(np.imag(val), 0.0))

        return grads

    def measure_energy(self, Ucirc):
        """
        Parameters
        ----------
        Ucirc : QuantumCircuit
            The state preparation circuit.
        """
        if self._fast:
            myQC = qforte.QuantumComputer(self._nqb)
            myQC.apply_circuit(Ucirc)
            val = np.real(myQC.direct_op_exp_val(self._qb_ham))
        else:
            Exp = qforte.Experiment(self._nqb, Ucirc, self._qb_ham, 2000)
            empty_params = []
            val = Exp.perfect_experimental_avg(empty_params)

        assert(np.isclose(np.imag(val),0.0))
        return val

    def energy_feval(self, params):
        Ucirc = self.build_Uvqc(params=params)
        # self._prev_energy = self._curr_energy
        Energy = self.measure_energy(Ucirc)

        # if(self._noise_factor > 1e-12):
        #     Energy = np.random.normal(Energy, self._noise_factor)

        self._curr_energy = Energy
        return Energy

    def gradient_ary_feval(self, params):
        grads = self.measure_gradient(params)

        if(self._noise_factor > 1e-14):
            noisy_grads = []
            for grad_m in grads:
                noisy_grads.append(np.random.normal(np.real(grad_m), self._noise_factor))
            grads = noisy_grads

        self._curr_grad_norm = np.linalg.norm(grads)
        self._grad_vec_evals += 1
        self._grad_m_evals += len(self._tamps)

        # iter_str = str(int(self._grad_vec_evals))
        # np.savetxt(f"qf_gsd_grad_{iter_str}.dat", np.real(grads))
        # np.savetxt(f"qf_gsd_params_{iter_str}.dat", np.real(params))

        return np.asarray(grads)

    # def callback(self, x):
    #     # print(f"\n -Minimum energy this iteration:    {self.energy_feval(x):+12.10f}")
    #
    #     self._k_counter += 1
    #
    #     if(self._k_counter == 1):
    #         # print(f'     {self._k_counter:7}        {self._curr_energy:+12.10f}       -----------      {self._grad_vec_evals:4}         {self._curr_grad_norm:+12.10f}')
    #         print('\n    k iteration         Energy               dE           Ngvec ev      Ngm ev*         ||g||')
    #         print('--------------------------------------------------------------------------------------------------')
    #         if (self._print_summary_file):
    #             f = open("summary.dat", "w+", buffering=1)
    #             f.write('\n#    k iteration         Energy               dE           Ngvec ev      Ngm ev*         ||g||')
    #             f.write('\n#--------------------------------------------------------------------------------------------------')
    #             f.close()
    #
    #     # else:
    #     dE = self._curr_energy - self._prev_energy
    #     print(f'     {self._k_counter:7}        {self._curr_energy:+12.10f}      {dE:+12.10f}      {self._grad_vec_evals:4}        {self._grad_m_evals:6}       {self._curr_grad_norm:+12.10f}')
    #
    #     if (self._print_summary_file):
    #         f = open("summary.dat", "a", buffering=1)
    #         f.write(f'\n       {self._k_counter:7}        {self._curr_energy:+12.12f}      {dE:+12.12f}      {self._grad_vec_evals:4}        {self._grad_m_evals:6}       {self._curr_grad_norm:+12.12f}')
    #         f.close()
    #
    #     self._prev_energy = self._curr_energy

    def verify_required_UCCPQE_attributes(self):
        # if self._use_analytic_grad is None:
        #     raise NotImplementedError('Concrete UCCVQE class must define self._use_analytic_grad attribute.')

        if not hasattr(self, '_pool_type'):
            raise NotImplementedError('Concrete UCCVQE class must define self._pool_type attribute.')

        if not hasattr(self, '_pool_obj'):
            raise NotImplementedError('Concrete UCCVQE class must define self._pool_obj attribute.')
