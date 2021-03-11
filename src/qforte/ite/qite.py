import qforte as qf
from qforte.abc.algorithm import Algorithm
from qforte.utils.transforms import (get_jw_organizer,
                                    organizer_to_circuit)

from qforte.utils.state_prep import *
from qforte.utils.trotterization import trotterize
from qforte.helper.printing import *
import copy
import numpy as np
from scipy.linalg import lstsq

class QITE(Algorithm):
    def run(self,
            beta=1.0,
            db=0.2,
            expansion_type='SD',
            sparseSb=True,
            b_thresh=1.0e-6,
            x_thresh=1.0e-8):

        self._beta = beta
        self._db = db
        self._nbeta = int(beta/db)+1
        self._sq_ham = self._sys.get_sq_hamiltonian()
        self._expansion_type = expansion_type
        self._sparseSb = sparseSb
        self._total_phase = 1.0 + 0.0j
        self._Uqite = qf.QuantumCircuit()
        self._b_thresh = b_thresh
        self._x_thresh = x_thresh

        self._n_classical_params = 0
        self._n_cnot = 0
        self._n_pauli_trm_measures = 0

        qc_ref = qf.QuantumComputer(self._nqb)
        qc_ref.apply_circuit(self._Uprep)
        self._Ekb = [np.real(qc_ref.direct_op_exp_val(self._qb_ham))]

        # Print options banner (should done for all algorithms).
        self.print_options_banner()

        # Build expansion pool.
        self.build_expansion_pool2()

        # print('\n sig op pool \n')
        #
        #    # qop
        # for term in self._sig.terms():
        #     qf.smart_print(term[1])
        #         # is a circ
        #     for circ in term[1].terms():
        #         print(circ[1])
        #         for gate in circ[1].gates():
        #             print(gate.gate_id(), gate.target())
        #
        #
        # print('\n\n')

        # Do the imaginary time evolution.
        self.evolve()

        # Print summary banner (should done for all algorithms).
        self.print_summary_banner()

        # verify that required attributes were defined
        # (should done for all algorithms).
        self.verify_run()

    def run_realistic(self):
        raise NotImplementedError('run_realistic() is not yet implemented for QITE.')

    def verify_run(self):
        self.verify_required_attributes()

    def print_options_banner(self):
        print('\n-----------------------------------------------------')
        print('     Quantum Imaginary Time Evolution Algorithm   ')
        print('-----------------------------------------------------')

        print('\n\n                 ==> QITE options <==')
        print('-----------------------------------------------------------')
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

        # Specific QITE options.
        print('Total imaginary evolution time (beta):   ',  self._beta)
        print('Imaginary time step (db):                ',  self._db)
        print('Expansion type:                          ',  self._expansion_type)
        print('x value threshold:                       ',  self._x_thresh)
        print('Use sparse tensors to solve Sx = b:      ',  str(self._sparseSb))
        if(self._sparseSb):
            print('b value threshold:                       ',  str(self._b_thresh))
        print('\n')

    def print_summary_banner(self):
        print('\n\n                        ==> QITE summary <==')
        print('-----------------------------------------------------------')
        print('Final QITE Energy:                        ', round(self._Egs, 10))
        print('Number of operators in pool:              ', self._NI)
        print('Number of classical parameters used:      ', self._n_classical_params)
        print('Number of CNOT gates in deepest circuit:  ', self._n_cnot)
        print('Number of Pauli term measurements:        ', self._n_pauli_trm_measures)

    def build_expansion_pool(self):
        print('\n==> Building expansion pool <==')
        self._sig = qf.QuantumOpPool()

        if(self._expansion_type == 'complete_qubit'):
            if (self._nqb > 6):
                raise ValueError('Using complete qubits expansion will reslut in a very large number of terms!')
            self._sig.fill_pool("complete_qubit", self._ref)

        elif(self._expansion_type == 'cqoy'):
            self._sig.fill_pool("cqoy", self._ref)

        elif(self._expansion_type == 'qbGSD'):
            # TODO (opt), put this on C side
            op_organizer = get_jw_organizer(self._sq_ham, combine=False)
            uniqe_org = []
            for term in op_organizer:
                for coeff, word in term:
                    nygates = 0
                    for pgate in word:
                        if pgate[0] == 'Y':
                            nygates += 1
                    if (nygates%2 != 0):
                        uniqe_term = [1.0, word]
                        if(uniqe_term not in uniqe_org):
                            uniqe_org.append(uniqe_term)
                            self._sig.add_term(1.0, organizer_to_circuit([uniqe_term]))

        elif(self._expansion_type == 'test'):
            self._sig.fill_pool("test", self._ref)

        else:
            raise ValueError('Invalid expansion type specified.')

        self._NI = len(self._sig.terms())

    def build_expansion_pool2(self):
        print('\n==> Building expansion pool <==')
        self._sig = qf.QuantumOpPool()

        if(self._expansion_type == 'complete_qubit'):
            if (self._nqb > 6):
                raise ValueError('Using complete qubits expansion will reslut in a very large number of terms!')
            self._sig.fill_pool("complete_qubit", self._ref)

        elif(self._expansion_type == 'cqoy'):
            self._sig.fill_pool("cqoy", self._ref)

        elif(self._expansion_type == 'ham'):
            # TODO (opt), put this on C side
            op_organizer = get_jw_organizer(self._sq_ham, combine=False)
            uniqe_org = []
            for term in op_organizer:
                for coeff, word in term:
                    nygates = 0
                    for pgate in word:
                        if pgate[0] == 'Y':
                            nygates += 1
                    if (nygates%2 != 0):
                        uniqe_term = [1.0, word]
                        if(uniqe_term not in uniqe_org):
                            uniqe_org.append(uniqe_term)
                            self._sig.add_term(1.0, organizer_to_circuit([uniqe_term]))

        elif(self._expansion_type == 'SD' or 'GSD' or 'SDT' or 'SDTQ' or 'SDTQP' or 'SDTQPH'):
            P = qf.SQOpPool()
            P.set_orb_spaces(self._ref)
            P.fill_pool(self._expansion_type)
            sig_temp = P.get_quantum_operator("commuting_grp_lex", False)
            # qf.smart_print(sig_temp)

            for alph, rho in sig_temp.terms():
                nygates = 0
                temp_rho = qf.QuantumCircuit()
                for gate in rho.gates():
                    temp_rho.add_gate(qf.make_gate(gate.gate_id(), gate.target(), gate.control()))
                    if (gate.gate_id() == "Y"):
                        nygates += 1

                if (nygates%2 != 0):
                    rho_op = qf.QuantumOperator()
                    rho_op.add_term(1.0, temp_rho)
                    self._sig.add_term(1.0, rho_op)


        # elif(self._expansion_type == 'qbGSD'):
        #     # TODO (opt), put this on C side
        #     op_organizer = get_jw_organizer(self._sq_ham, combine=False)
        #     uniqe_org = []
        #     for term in op_organizer:
        #         for coeff, word in term:
        #             nygates = 0
        #             for pgate in word:
        #                 if pgate[0] == 'Y':
        #                     nygates += 1
        #             if (nygates%2 != 0):
        #                 uniqe_term = [1.0, word]
        #                 if(uniqe_term not in uniqe_org):
        #                     uniqe_org.append(uniqe_term)
        #                     self._sig.add_term(1.0, organizer_to_circuit([uniqe_term]))

        elif(self._expansion_type == 'test'):
            self._sig.fill_pool("test", self._ref)

        else:
            raise ValueError('Invalid expansion type specified.')

        self._NI = len(self._sig.terms())


    def build_S(self):
        Idim = self._NI

        S = np.zeros((Idim, Idim), dtype=complex)

        Ipsi_qc = qf.QuantumComputer(self._nqb)
        Ipsi_qc.set_coeff_vec(copy.deepcopy(self._qc.get_coeff_vec()))
        CI = np.zeros(shape=(Idim, int(2**self._nqb)), dtype=complex)

        for i in range(1, Idim):
            S[i-1][i-1] = 1.0
            Ipsi_qc.apply_operator(self._sig.terms()[i][1])
            CI[i,:] = copy.deepcopy(Ipsi_qc.get_coeff_vec())
            for j in range(1, i):
                val = np.vdot(CI[i,:], CI[j,:])
                S[i][j] = val
                S[j][i] = val

            Ipsi_qc.set_coeff_vec(copy.deepcopy(self._qc.get_coeff_vec()))

        return np.real(S)


    def build_sparse_S_b(self, b):
        b_sparse = []
        idx_sparse = []
        for I, bI in enumerate(b):
            if(np.abs(bI) > self._b_thresh):
                idx_sparse.append(I)
                b_sparse.append(bI)

        Idim = len(idx_sparse)
        self._n_pauli_trm_measures += int(Idim*(Idim+1)*0.5)

        S = np.zeros((len(b_sparse),len(b_sparse)), dtype=complex)

        Ipsi_qc = qf.QuantumComputer(self._nqb)
        Ipsi_qc.set_coeff_vec(copy.deepcopy(self._qc.get_coeff_vec()))
        CI = np.zeros(shape=(Idim, int(2**self._nqb)), dtype=complex)

        for i in range(1, Idim):
            S[i-1][i-1] = 1.0
            Ii = idx_sparse[i]
            Ipsi_qc.apply_operator(self._sig.terms()[Ii][1])
            CI[i,:] = copy.deepcopy(Ipsi_qc.get_coeff_vec())
            for j in range(1, i):
                val = np.vdot(CI[i,:], CI[j,:])
                S[i][j] = val
                S[j][i] = val

            Ipsi_qc.set_coeff_vec(copy.deepcopy(self._qc.get_coeff_vec()))

        return idx_sparse, np.real(S), np.real(b_sparse)

    def build_b(self):

        b  = np.zeros(self._NI, dtype=complex)

        # Now uses normalization for all H rather than for each term Hl.
        term = np.sqrt(1 - 2*self._db*self._Ekb[-1])
        fo = -1.0j / term

        self._n_pauli_trm_measures += self._Nl * self._NI

        self._Hpsi_qc = qf.QuantumComputer(self._nqb)
        self._Hpsi_qc.set_coeff_vec(copy.deepcopy(self._qc.get_coeff_vec()))
        self._Hpsi_qc.apply_operator(self._qb_ham)
        C_Hpsi_qc = copy.deepcopy(self._Hpsi_qc.get_coeff_vec())

        for I in range(self._NI):
            self._Hpsi_qc.apply_operator(self._sig.terms()[I][1])
            val = np.vdot(self._qc.get_coeff_vec(), self._Hpsi_qc.get_coeff_vec())
            b[I] = self._sig.terms()[I][0] * val * fo
            self._Hpsi_qc.set_coeff_vec(C_Hpsi_qc)

        return np.real(b)

    def do_quite_step(self):

        btot = self.build_b()
        A = qf.QuantumOperator()

        if(self._sparseSb):
            sp_idxs, S, btot = self.build_sparse_S_b(btot)
        else:
            S = self.build_S()

        x = lstsq(S, btot)[0]
        x = np.real(x)

        if(self._sparseSb):
            for I, spI in enumerate(sp_idxs):
                if np.abs(x[I]) > self._x_thresh:
                    A.add_term(-1.0j * self._db * x[I], self._sig.terms()[spI][1].terms()[0][1])
                    self._n_classical_params += 1

        else:
            for I, SigI in enumerate(self._sig.terms()):
                if np.abs(x[I]) > self._x_thresh:
                    A.add_term(-1.0j * self._db * x[I], SigI[1].terms()[0][1])
                    self._n_classical_params += 1

        if(self._verbose):
            print('\nbtot:\n ', btot)
            print('\n S:  \n')
            matprint(S)
            print('\n x:  \n')
            print(x)

        eiA_kb, phase1 = trotterize(A, trotter_number=self._trotter_number)
        self._total_phase *= phase1
        self._Uqite.add_circuit(eiA_kb)
        self._qc.apply_circuit(eiA_kb)
        self._Ekb.append(np.real(self._qc.direct_op_exp_val(self._qb_ham)))

        self._n_cnot += eiA_kb.get_num_cnots()

        if(self._verbose):
            qf.smart_print(self._qc)

    def evolve(self):
        self._Uqite.add_circuit(self._Uprep)
        self._qc = qf.QuantumComputer(self._nqb)
        self._qc.apply_circuit(self._Uqite)

        print(f"{'beta':>7}{'E(beta)':>18}{'N(params)':>14}{'N(CNOT)':>18}{'N(measure)':>20}")
        print('-------------------------------------------------------------------------------')
        print(f' {0.0:7.3f}    {self._Ekb[0]:+15.9f}    {self._n_classical_params:8}        {self._n_cnot:10}        {self._n_pauli_trm_measures:12}')

        if (self._print_summary_file):
            f = open("summary.dat", "w+", buffering=1)
            f.write(f"#{'beta':>7}{'E(beta)':>18}{'N(params)':>14}{'N(CNOT)':>18}{'N(measure)':>20}\n")
            f.write('#-------------------------------------------------------------------------------\n')
            f.write(f'  {0.0:7.3f}    {self._Ekb[0]:+15.9f}    {self._n_classical_params:8}        {self._n_cnot:10}        {self._n_pauli_trm_measures:12}\n')

        for kb in range(1, self._nbeta):
            self.do_quite_step()
            print(f' {kb*self._db:7.3f}    {self._Ekb[kb]:+15.9f}    {self._n_classical_params:8}        {self._n_cnot:10}        {self._n_pauli_trm_measures:12}')
            if (self._print_summary_file):
                f.write(f'  {kb*self._db:7.3f}    {self._Ekb[kb]:+15.9f}    {self._n_classical_params:8}        {self._n_cnot:10}        {self._n_pauli_trm_measures:12}\n')
        self._Egs = self._Ekb[-1]

        if (self._print_summary_file):
            f.close()

    def print_expansion_ops(self):
        print('\nQITE expansion operators:')
        print('-------------------------')
        print(self._sig.str())
