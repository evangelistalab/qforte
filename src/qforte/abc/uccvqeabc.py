import qforte as qf
from abc import abstractmethod
from qforte.abc.vqeabc import VQE
from qforte.utils.op_pools import *

from qforte.experiment import *
from qforte.utils.transforms import *
from qforte.utils.state_prep import ref_to_basis_idx
from qforte.utils.trotterization import trotterize

class UCCVQE(VQE):

    @abstractmethod
    def get_num_ham_measurements(self):
        pass

    @abstractmethod
    def get_num_comut_measurements(self):
        pass

    def fill_pool(self):

        self._pool_obj = qf.SQOpPool()
        self._pool_obj.set_orb_spaces(self._ref)

        if (self._pool_type=='sa_SD'):
            self._pool_obj.fill_pool('sa_SD')
        elif (self._pool_type=='SD'):
            self._pool_obj.fill_pool('SD')
        elif (self._pool_type=='SDT'):
            self._pool_obj.fill_pool('SDT')
        elif (self._pool_type=='SDTQ'):
            self._pool_obj.fill_pool('SDTQ')
        elif (self._pool_type=='SDTQP'):
            self._pool_obj.fill_pool('SDTQP')
        elif (self._pool_type=='SDTQPH'):
            self._pool_obj.fill_pool('SDTQPH')
        else:
            raise ValueError('Invalid operator pool type specified.')

        self._pool = self._pool_obj.terms()

        self._Nm = []
        for m in range(len(self._pool_obj.terms())):
            self._Nm.append(len(self._pool_obj.terms()[m][1].jw_transform().terms()))

    def fill_comutator_pool(self):
        print('\n\n==> Building comutator pool for gradient measurement.')
        self._comutator_pool = self._pool_obj.get_quantum_op_pool()
        self._comutator_pool.join_as_comutator(self._qb_ham)
        print('==> Comutator pool construction complete.')

    # TODO (opt major): write a C function that prepares this super efficiently
    def build_Uvqc(self, params=None):
        """ This function returns the QuantumCircuit object built
        from the appropiate ampltudes (tops)

        Parameters
        ----------
        params : list
            A lsit of parameters define the variational degress of freedom in
            the state perparation circuit Uvqc.
        """
        temp_pool = qf.SQOpPool()
        if params is None:
            for tamp, top in zip(self._tamps, self._tops):
                temp_pool.add_term(tamp, self._pool[top][1])
        else:
            for param, top in zip(params, self._tops):
                temp_pool.add_term(param, self._pool[top][1])

        A = temp_pool.get_quantum_operator('comuting_grp_lex')

        U, phase1 = trotterize(A, trotter_number=self._trotter_number)
        Uvqc = qforte.QuantumCircuit()
        Uvqc.add_circuit(self._Uprep)
        Uvqc.add_circuit(U)
        if phase1 != 1.0 + 0.0j:
            raise ValueError("Encountered phase change, phase not equal to (1.0 + 0.0i)")

        return Uvqc

    def measure_comutator_gradient(self, HAm, Ucirc, idxs=[], params=None):
        """
        Parameters
        ----------
        HAm : QuantumOpPool
            The comutator to measure.

        Ucirc : QuantumCircuit
            The state preparation circuit.
        """

        if self._fast:
            myQC = qforte.QuantumComputer(self._nqb)
            myQC.apply_circuit(Ucirc)
            if(len(idxs)==0):
                grads = myQC.direct_oppl_exp_val(HAm)
            else:
                grads = myQC.direct_idxd_oppl_exp_val(HAm, idxs)

        else:
            pass
            # TODO (cleanup): remove N_samples as argument (implement variance based thresh)
            # TODO: need to implement this as a for loop over terms in QuantumOpPool
            # Exp = qforte.Experiment(self._nqb, Ucirc, HAm, 1000)
            # empty_params = []
            # val = Exp.perfect_experimental_avg(empty_params)
        for val in grads:
            assert(np.isclose(np.imag(val), 0.0))

        return np.real(grads)

    def measure_gradient(self, params=None):
        """
        Parameters
        ----------
        HAm : QuantumOpPool
            The comutator to measure.

        Ucirc : QuantumCircuit
            The state preparation circuit.
        """

        if self._fast==False:
            raise ValueError("self._fast must be True for gradient measurement.")

        grads = []
        M = len(self._tamps)
        omega_o = ref_to_basis_idx(self._ref)

        for mu in range(M):
            lo_pool = qf.SQOpPool()
            hi_pool = qf.SQOpPool()
            if params is None:
                for nu in range(mu):
                    # nu indexes term THAT ARE ALREDY INCLUDED from the pool
                    tamp = self._tamps[nu] # actual amp t_nu
                    top = self._tops[nu] # actual pool idx for K_nu
                    lo_pool.add_term(tamp, self._pool[top][1])

                for nu in range(mu, M):
                    tamp = self._tamps[nu]
                    top = self._tops[nu]
                    hi_pool.add_term(tamp, self._pool[top][1])


            else:
                for nu in range(mu):
                    tamp = params[nu]
                    top = self._tops[nu]
                    lo_pool.add_term(tamp, self._pool[top][1])

                for nu in range(mu, M):
                    tamp = params[nu]
                    top = self._tops[nu]
                    hi_pool.add_term(tamp, self._pool[top][1])

            Kmu = self._pool[self._tops[mu]][1].jw_transform()
            Kmu.mult_coeffs(self._pool[self._tops[mu]][0])

            Alo = lo_pool.get_quantum_operator('comuting_grp_lex')
            Ahi = hi_pool.get_quantum_operator('comuting_grp_lex')

            Ulo, plo = trotterize(Alo, trotter_number=self._trotter_number)
            Uhi, phi = trotterize(Ahi, trotter_number=self._trotter_number)
            if (plo != 1.0 + 0.0j) or (phi != 1.0 + 0.0j):
                raise ValueError("Encountered phase change, phase not equal to (1.0 + 0.0i)")

            qc = qforte.QuantumComputer(self._nqb)
            qc.apply_circuit(self._Uprep)
            qc.apply_circuit(Ulo)
            qc.apply_operator(Kmu)
            qc.apply_circuit(Uhi)
            qc.apply_operator(self._qb_ham)
            qc.apply_circuit(Uhi.adjoint())
            qc.apply_circuit(Ulo.adjoint())

            grads.append( 2.0*qc.get_coeff_vec()[omega_o] )


        for val in grads:
            assert(np.isclose(np.imag(val), 0.0))

        return np.real(grads)

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
        return self.measure_energy(Ucirc)

    def gradient_ary_feval(self, params):
        # if self._use_htest_gradient:
        grads = self.measure_gradient(params)
        # else:
        # Uvqc = self.build_Uvqc(params=params)
        # grads = self.measure_comutator_gradient(self._comutator_pool, Uvqc, self._tops)

        return np.asarray(grads)

    def callback(self, x):
        print(f" -Minimum energy this iteration:    {self.energy_feval(x):+12.10f}")

    def verify_required_UCCVQE_attributes(self):
        if self._use_analytic_grad is None:
            raise NotImplementedError('Concrete UCCVQE class must define self._use_analytic_grad attribute.')

        if self._pool_type is None:
            raise NotImplementedError('Concrete UCCVQE class must define self._pool_type attribute.')

        if self._pool_obj is None:
            raise NotImplementedError('Concrete UCCVQE class must define self._pool_obj attribute.')
