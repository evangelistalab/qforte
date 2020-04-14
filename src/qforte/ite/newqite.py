import qforte
from qforte.abc.algorithm import Algorithm
from qforte.utils.transforms import (circuit_to_organizer,
                                    organizer_to_circuit,
                                    join_organizers,
                                    get_jw_organizer)

from qforte.utils.state_prep import *
from qforte.utils.trotterization import trotterize

import numpy as np
from scipy.linalg import lstsq

class QITE(Algorithm):
    def run(self,
            beta=1.0,
            db=0.2,
            expansion_type='qbGSD',
            sparseSb=True):

        self._beta = beta
        self._db = db
        self._nbeta = int(beta/db)+1
        self._sq_ham = self._sys.get_sq_hamiltonian()
        self._expansion_type = expansion_type
        self._sparseSb = sparseSb
        self._total_phase = 1.0 + 0.0j
        self._Uqite = qforte.QuantumCircuit()

        qc_ref = qforte.QuantumComputer(self._nqb)
        qc_ref.apply_circuit(self._Uprep)
        self._Ekb = [np.real(qc_ref.direct_op_exp_val(self._qb_ham))]

        # Print options banner (should done for all algorithms).
        self.print_options_banner()

        # Build expansion pool.
        self.build_expansion_pool()

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
        print('Use sparse tensors to solve Sx = b:      ',  str(self._sparseSb))
        print('\n')

    def print_summary_banner(self):
        print('\n\n                        ==> QITE summary <==')
        print('-----------------------------------------------------------')
        print('Final QITE Energy:                        ', round(self._Egs, 10))

    def to_base4(self, I):
        convert_str = "0123456789"
        if I < 4:
            return convert_str[I]
        else:
            return self.to_base4(I//4) + convert_str[I%4]

    def pauli_idx_str(self, I_str):
        res = ''
        for i in range(self._nqb-len(I_str)):
            res += '0'
        return res + I_str

    def build_expansion_pool(self):
        print('\n==> Building expansion pool <==')
        self._expansion_ops = []

        if(self._expansion_type == 'sqGSD'):
            SigI_org = []
            op_organizer = get_jw_organizer(self._sq_ham, combine=True)
            for l in range(len(op_organizer)):
                op_organizer[l][0] = 1.0 + 0.0j
                SigI_org.append([op_organizer[l]])

        elif(self._expansion_type == 'qbGSD'):
            op_organizer = get_jw_organizer(self._sq_ham, combine=False)
            uniqe_org = []
            SigI_org = []
            for term in op_organizer:
                for coeff, word in term:
                    uniqe_term = [1.0, word]
                    if(uniqe_term not in uniqe_org):
                        uniqe_org.append(uniqe_term)
                        SigI_org.append([uniqe_term])

        elif(self._expansion_type == 'complete_qubit'):
            if (self._nqb > 6):
                raise ValueError('Using complete qubits expansion will reslut in a very large number of terms!')

            SigI_org = []
            paulis = ['I', 'X', 'Y', 'Z']
            for I in range(4**(self._nqb)):
                paulistr = self.pauli_idx_str(self.to_base4(I))
                AI = []
                for k, gate_id in enumerate(paulistr):
                    if(gate_id != '0'):
                        AI.append( ( paulis[int(gate_id)], k ) )

                SigI_org.append([[1.0, AI]])

        else:
            raise ValueError('Invalid expansion type specified.')

        H_org = circuit_to_organizer(self._qb_ham)
        self._Nl = len(self._qb_ham.terms())
        self._NI = len(SigI_org)
        self._H = np.empty(shape=self._Nl, dtype=object)
        self._Sig = np.empty(shape=self._NI, dtype=object)
        self._HSig = np.empty(shape=(self._Nl, self._NI), dtype=object)
        self._SigSig = np.empty(shape=(self._NI, self._NI), dtype=object)

        for l, Hlorg in enumerate(H_org):
            self._H[l] = (organizer_to_circuit([Hlorg]))

        for I, Iorg in enumerate(SigI_org):
            self._Sig[I] = organizer_to_circuit(Iorg)
            for l, Hlorg in enumerate(H_org):
                sigI_Hl_org = join_organizers(Iorg, [Hlorg])
                self._HSig[l][I] = organizer_to_circuit(sigI_Hl_org)

            for J, Jorg in enumerate(SigI_org):
                sigI_sigJ_org = join_organizers(Iorg, Jorg)
                self._SigSig[I][J] = organizer_to_circuit(sigI_sigJ_org)

        print('\n      Expansion pool successfully built!\n')

    def build_S(self):
        S = np.empty((self._NI,self._NI), dtype=complex)
        qc = qforte.QuantumComputer(self._nqb)
        qc.apply_circuit(self._Uqite)
        # TODO: use only upper triangle of Mat (Nick)
        for I in range(self._NI):
            for J in range(self._NI):
                val = qc.direct_op_exp_val(self._SigSig[I][J])
                S[I][J] = val

        return np.real(S)

    def build_sparse_S_b(self, b):
        b_sparse = []
        idx_sparse = []
        for I, bI in enumerate(b):
            if(np.abs(bI) > 1e-6):
                idx_sparse.append(I)
                b_sparse.append(bI)

        S = np.empty((len(b_sparse),len(b_sparse)), dtype=complex)
        qc = qforte.QuantumComputer(self._nqb)
        qc.apply_circuit(self._Uqite)
        # TODO: use only upper triangle of Mat (Nick)
        for i, I in enumerate(idx_sparse):
            for j, J in enumerate(idx_sparse):
                val = qc.direct_op_exp_val(self._SigSig[I][J])
                S[i][j] = val

        return idx_sparse, np.real(S), np.real(b_sparse)

    def build_bl(self, l, cl):
        bo = -(1.0j / np.sqrt(cl))
        bl = np.zeros(shape=self._NI, dtype=complex)
        qc = qforte.QuantumComputer(self._nqb)
        qc.apply_circuit(self._Uqite)
        for I in range(self._NI):
            bl[I] = bo * qc.direct_op_exp_val(self._HSig[l][I])

        return np.real(bl)

    def do_quite_step(self):
        qc = qforte.QuantumComputer(self._nqb)
        qc.apply_circuit(self._Uqite)

        # Can take this approach if all Hl have same expansion terms
        btot = np.zeros(shape=self._NI, dtype=complex)
        for l, Hl in enumerate(self._H):
            cl = 1 - 2*self._db*qc.direct_op_exp_val(Hl)
            bl = self.build_bl(l, cl)
            btot = np.add(btot, bl)
            if(self._verbose):
                print('\n\nl:  ', l)
                print('\nbl: ', bl)

        A = qforte.QuantumOperator()

        if(self._sparseSb):
            sp_idxs, S, btot = self.build_sparse_S_b(btot)

        else:
            S = self.build_S()

        x = lstsq(S, btot)[0]
        x = np.real(x)

        if(self._sparseSb):
            for I, spI in enumerate(sp_idxs):
                if np.abs(x[I]) > 1e-6:
                    A.add_term(-1.0j * self._db * x[I], self._Sig[spI].terms()[0][1])

        else:
            for I, SigI in enumerate(self._Sig):
                if np.abs(x[I]) > 1e-6:
                    A.add_term(-1.0j * self._db * x[I], SigI.terms()[0][1])

        if(self._verbose):
            print('\nbtot:\n ', btot)

        eiA_kb, phase1 = trotterize(A, trotter_number=self._trotter_number)
        self._total_phase *= phase1
        self._Uqite.add_circuit(eiA_kb)

        qc.apply_circuit(eiA_kb)
        self._Ekb.append(np.real(qc.direct_op_exp_val(self._qb_ham)))

        if(self._verbose):
            qforte.smart_print(qc)

    def evolve(self):
        self._Uqite.add_circuit(self._Uprep)
        print(' Beta        <Psi_b|H|Psi_b> ')
        print('---------------------------------------')
        print(' ', round(0.00, 3), '       ', np.round(self._Ekb[0], 10))

        for kb in range(1, self._nbeta):
            self.do_quite_step()
            print(' ', round(kb*self._db, 3), '       ', np.round(self._Ekb[kb], 10))

        self._Egs = self._Ekb[-1]

    def print_expansion_ops(self):
        print('\nQITE expansion operators:')
        print('-------------------------')
        for I, SigI in enumerate(self._Sig):
            print('\nI: ', I)
            qforte.smart_print(SigI)
