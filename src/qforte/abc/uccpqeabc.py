import qforte as qf
from abc import abstractmethod
from qforte.abc.pqeabc import PQE
from qforte.abc.ansatz import UCC
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

        self._Nm = [len(operator.jw_transform().terms()) for _, operator in self._pool_obj.terms()]

    # TODO (opt major): write a C function that prepares this super efficiently
    def build_Uvqc(self, amplitudes=None):
        """ This function returns the QuantumCircuit object built
        from the appropiate amplitudes (tops)

        Parameters
        ----------
        amplitudes : list
            A list of parameters that define the variational degrees of freedom in
            the state preparation circuit Uvqc. This is needed for the scipy minimizer.
        """

        ansatz = UCC(self._trotter_number, self._tamps, self._tops, self._pool)
        U = ansatz.ansatz_circuit(amplitudes)

        Uvqc = qforte.QuantumCircuit()
        Uvqc.add_circuit(self._Uprep)
        Uvqc.add_circuit(U)

        return Uvqc

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

    def verify_required_UCCPQE_attributes(self):

        if not hasattr(self, '_pool_type'):
            raise NotImplementedError('Concrete UCCVQE class must define self._pool_type attribute.')

        if not hasattr(self, '_pool_obj'):
            raise NotImplementedError('Concrete UCCVQE class must define self._pool_obj attribute.')
