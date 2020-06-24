import qforte as qf
from abc import abstractmethod
from qforte.abc.vqeabc import VQE
from qforte.utils.op_pools import *

from qforte.experiment import *
from qforte.utils.transforms import *
from qforte.utils.trotterization import trotterize

class UCCVQE2(VQE):

    @abstractmethod
    def get_num_ham_measurements(self):
        pass

    @abstractmethod
    def get_num_comut_measurements(self):
        pass

    # TODO (cleanup): rename functin 'build_pool'
    def fill_pool(self):
        # TODO: change "SD" to "sa_SD"
        if (self._pool_type=='SD'):
            #new
            self._pool_obj = qf.SQOpPool()
            self._pool_obj.set_orb_spaces(self._ref)
            self._pool_obj.fill_pool('sa_SD')
        else:
            raise ValueError('Invalid operator pool type specified.')

        # self._pool_obj.fill_pool()
        # TODO (opt): rename _pool to '_pool_lst' and conver to numpy array.
        self._pool = self._pool_obj.terms() # may be uneccecary
        # help(self._pool)

    def fill_comutator_pool(self):
        print('\n\n==> Building comutator pool for gradient measurement.')
        # TODO (opt): Perhaps after moving operator handling to to C side.


        self._comutator_pool = self._pool_obj.get_quantum_op_pool()
        self._comutator_pool.join_as_comutator(self._qb_ham)


        # for i in range(len(self._pool)):
            # Am_org = get_ucc_jw_organizer(self._pool[i], already_anti_herm=True)
            # H_org = circuit_to_organizer(self._qb_ham)
            # HAm_org = join_H_Am_organizers(H_org, Am_org)
            # HAm = organizer_to_circuit(HAm_org) # actually returns a single-term QuantumOperator
            # self._comutator_pool.append(HAm)
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
        # sq_ops = []
        # for j, pool_idx in enumerate(self._tops):
        #     appended_term = copy.deepcopy(self._pool[pool_idx])
        #     for l in range(len(appended_term)):
        #         if params is None:
        #             appended_term[l][1] *= self._tamps[j]
        #         else:
        #             appended_term[l][1] *= params[j]
        #         sq_ops.append(appended_term[l])
        #
        # Uorg = get_ucc_jw_organizer(sq_ops, already_anti_herm=True)
        # A = organizer_to_circuit(Uorg)

        #new#<
        # need to build new SQOpPool object,
        temp_pool = qf.SQOpPool()
        # and fill with correct amps
        if params is None:
            for tamp, top in zip(self._tamps, self._tops):
                temp_pool.add_term(tamp, self._pool[top][1])
        # or with params
        else:
            for param, top in zip(params, self._tops):
                # print(type(param))
                # print(type(self._pool[top]))
                temp_pool.add_term(param, self._pool[top][1])

        A = temp_pool.get_quantum_operator()
        A.order_terms()
        #new#>

        U, phase1 = trotterize(A, trotter_number=self._trotter_number)
        Uvqc = qforte.QuantumCircuit()
        Uvqc.add_circuit(self._Uprep)
        Uvqc.add_circuit(U)
        if phase1 != 1.0 + 0.0j:
            raise ValueError("Encountered phase change, phase not equal to (1.0 + 0.0i)")

        return Uvqc

    def measure_gradient(self, HAm, Ucirc, idxs=[]):
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
            Exp = qforte.Experiment(self._nqb, Ucirc, self._qb_ham, 1000)
            empty_params = []
            val = Exp.perfect_experimental_avg(empty_params)

        assert(np.isclose(np.imag(val),0.0))
        return val

    def energy_feval(self, params):
        Ucirc = self.build_Uvqc(params=params)
        return self.measure_energy(Ucirc)

    def gradient_ary_feval(self, params):
        Uvqc = self.build_Uvqc(params=params)
        grads = self.measure_gradient(self._comutator_pool, Uvqc, self._tops)
        # grad_lst = []
        # for m in self._tops:
        #     grad_lst.append(self.measure_gradient(self._comutator_pool[m], Uvqc))

        return np.asarray(grads)

    def verify_required_UCCVQE_attributes(self):
        if self._use_analytic_grad is None:
            raise NotImplementedError('Concrete UCCVQE class must define self._use_analytic_grad attribute.')

        if self._pool_type is None:
            raise NotImplementedError('Concrete UCCVQE class must define self._pool_type attribute.')

        if self._pool_obj is None:
            raise NotImplementedError('Concrete UCCVQE class must define self._pool_obj attribute.')
