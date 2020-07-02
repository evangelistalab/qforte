import qforte as qf
from qforte.abc.algorithm import Algorithm
from qforte.utils.transforms import (circuit_to_organizer,
                                    organizer_to_circuit,
                                    join_organizers,
                                    get_jw_organizer)

from qforte.utils.state_prep import *
from qforte.utils.trotterization import trotterize
from qforte.helper.printing import *

import numpy as np
from scipy.linalg import lstsq

class QITE(Algorithm):
    def run(self,
            beta=1.0,
            db=0.2,
            expansion_type='qbGSD',
            sparseSb=True,
            version=1,
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

        qc_ref = qf.QuantumComputer(self._nqb)
        qc_ref.apply_circuit(self._Uprep)
        self._Ekb = [np.real(qc_ref.direct_op_exp_val(self._qb_ham))]

        # Print options banner (should done for all algorithms).
        self.print_options_banner()

        if (version==1):
            self.build_expansion_pool()
            self.evolve()

        if (version==2):
            # Build expansion pool.
            self.build_expansion_pool2()

            # Do the imaginary time evolution.
            self.evolve2()

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
        print('x value threshold:                       ',  self._x_thresh)
        print('Use sparse tensors to solve Sx = b:      ',  str(self._sparseSb))
        if(self._sparseSb):
            print('b value threshold:                       ',  str(self._b_thresh))
        print('\n')

    def print_summary_banner(self):
        print('\n\n                        ==> QITE summary <==')
        print('-----------------------------------------------------------')
        print('Final QITE Energy:                        ', round(self._Egs, 10))
        # dim of op space
        # nparams
        # nCNOT
        # nPauiMeasurements

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

        elif(self._expansion_type == 'cqoy'):
            if (self._nqb > 6):
                raise ValueError('Using complete qubits expansion will reslut in a very large number of terms!')

            SigI_org = []
            paulis = ['I', 'X', 'Y', 'Z']
            for I in range(4**(self._nqb)):
                paulistr = self.pauli_idx_str(self.to_base4(I))
                AI = []
                nygates = 0
                for k, gate_id in enumerate(paulistr):
                    if(gate_id == '2'):
                        nygates += 1
                    if(gate_id != '0'):
                        AI.append( ( paulis[int(gate_id)], k ) )

                if(nygates%2 != 0):
                    SigI_org.append([[1.0, AI]])

        elif(self._expansion_type == 'test'):
            SigI_org = []

            SigI_org.append([[1.0, [ ('X', 0), ('X', 1), ('X', 2), ('Y', 3) ]]])
            SigI_org.append([[1.0, [ ('X', 0), ('X', 1), ('Y', 2), ('X', 3) ]]])
            SigI_org.append([[1.0, [ ('X', 0), ('Y', 1), ('X', 2), ('X', 3) ]]])
            SigI_org.append([[1.0, [ ('X', 0), ('Y', 1), ('Y', 2), ('Y', 3) ]]])
            SigI_org.append([[1.0, [ ('Y', 0), ('X', 1), ('X', 2), ('X', 3) ]]])
            SigI_org.append([[1.0, [ ('Y', 0), ('X', 1), ('Y', 2), ('Y', 3) ]]])
            SigI_org.append([[1.0, [ ('Y', 0), ('Y', 1), ('X', 2), ('Y', 3) ]]])
            SigI_org.append([[1.0, [ ('Y', 0), ('Y', 1), ('Y', 2), ('X', 3) ]]])

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

    def build_expansion_pool2(self):
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
                    uniqe_term = [1.0, word]
                    if(uniqe_term not in uniqe_org):
                        uniqe_org.append(uniqe_term)
                        self._sig.add_term(1.0, organizer_to_circuit([uniqe_term]))

        elif(self._expansion_type == 'test'):
            self._sig.fill_pool("test", self._ref)

        else:
            raise ValueError('Invalid expansion type specified.')

        self._Nl = len(self._qb_ham.terms())
        self._NI = len(self._sig.terms())
        self._h = np.ones(self._Nl, dtype=complex)
        for l, term in enumerate(self._qb_ham.terms()):
            self._h[l] = term[0]

        # compare tranfering terms to using the fill_pool fn for speed
        self._sigH = qf.QuantumOpPool()
        self._sig2 = qf.QuantumOpPool()
        for term in self._sig.terms():
            self._sigH.add_term(term[0], term[1])
            self._sig2.add_term(term[0], term[1])

        self._sig2.square(True)
        self._sigH.join_op_from_right_lazy(self._qb_ham)

        print('\n      Expansion pool successfully built!\n')

    def build_S2(self):
        S = np.empty((self._NI,self._NI), dtype=complex)
        Svec = self._qc.direct_oppl_exp_val(self._sig2)
        for I in range(self._NI):
            for J in range(I, self._NI):
                K = I*self._NI - int(I*(I-1)/2) + (J-I)
                val = Svec[K]
                S[I][J] = val
                S[J][I] = val

        return np.real(S)

    def build_S(self):
        S = np.empty((self._NI,self._NI), dtype=complex)
        for I in range(self._NI):
            for J in range(I, self._NI):
                val = self._qc.direct_op_exp_val(self._SigSig[I][J])
                S[I][J] = val
                S[J][I] = val

        return np.real(S)

    def build_sparse_S_b(self, b):
        b_sparse = []
        idx_sparse = []
        for I, bI in enumerate(b):
            if(np.abs(bI) > self._b_thresh):
                idx_sparse.append(I)
                b_sparse.append(bI)

        S = np.empty((len(b_sparse),len(b_sparse)), dtype=complex)
        for i in range(len(idx_sparse)):
            for j in range(i,len(idx_sparse)):
                val = self._qc.direct_op_exp_val(self._SigSig[idx_sparse[i]][idx_sparse[j]])
                S[i][j] = val
                S[j][i] = val

        return idx_sparse, np.real(S), np.real(b_sparse)

    def build_sparse_S_b2(self, b):
        b_sparse = []
        idx_sparse = []
        K_idx_sparse = []
        for I, bI in enumerate(b):
            if(np.abs(bI) > self._b_thresh):
                idx_sparse.append(I)
                b_sparse.append(bI)

        for i in range(len(idx_sparse)):
            for j in range(i,len(idx_sparse)):
                k_sp = idx_sparse[i]*self._NI
                k_sp -= int( idx_sparse[i] * (idx_sparse[i]-1)/2 )
                k_sp += idx_sparse[j] - idx_sparse[i]
                K_idx_sparse.append(k_sp)

        S = np.empty((len(b_sparse),len(b_sparse)), dtype=complex)
        Svec = self._qc.direct_idxd_oppl_exp_val(self._sig2, K_idx_sparse)
        for i in range(len(idx_sparse)):
            for j in range(i,len(idx_sparse)):
                k = i*len(b_sparse) - int(i*(i-1)/2) + (j-i)
                val = Svec[k]
                S[i][j] = val
                S[j][i] = val

        return idx_sparse, np.real(S), np.real(b_sparse)

    def build_bl(self, l, cl):
        bo = -(1.0j / np.sqrt(cl))
        bl = np.zeros(shape=self._NI, dtype=complex)
        for I in range(self._NI):
            bl[I] = bo * self._qc.direct_op_exp_val(self._HSig[l][I])

        return np.real(bl)

    def build_b2(self):
        fo = np.zeros(self._Nl, dtype=complex)
        for l, Hl in enumerate(self._qb_ham.terms()):
            term = np.sqrt(1 - 2*self._db*Hl[0]*self._qc.direct_circ_exp_val(Hl[1]))
            fo[l] = -1.0j / term

        return np.real(self._qc.direct_oppl_exp_val_w_mults(self._sigH, fo))

    def do_quite_step(self):
        self._qc = qf.QuantumComputer(self._nqb)
        self._qc.apply_circuit(self._Uqite)
        btot = np.zeros(shape=self._NI)

        for l, Hl in enumerate(self._H):
            cl = 1 - 2*self._db*self._qc.direct_op_exp_val(Hl)
            bl = self.build_bl(l, cl)
            btot = np.add(btot, bl)

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
                    A.add_term(-1.0j * self._db * x[I], self._Sig[spI].terms()[0][1])

        else:
            for I, SigI in enumerate(self._Sig):
                if np.abs(x[I]) > self._x_thresh:
                    A.add_term(-1.0j * self._db * x[I], SigI.terms()[0][1])

        if(self._verbose):
            if(self._sparseSb):
                print('\nsp_idxs:\n ', sp_idxs)

            print('\nbo:\n ')
            for val in self._bo:
                print('  ', val)
            print('\nbtot:\n ', btot)
            print('\n S:  \n')
            matprint(S)
            print('\n x:  \n')
            print(x)

        # np.savetxt('b_v1.dat', btot)
        # np.savetxt('s_v1.dat', S)

        eiA_kb, phase1 = trotterize(A, trotter_number=self._trotter_number)
        self._total_phase *= phase1
        self._Uqite.add_circuit(eiA_kb)

        self._qc.apply_circuit(eiA_kb)
        self._Ekb.append(np.real(self._qc.direct_op_exp_val(self._qb_ham)))

        if(self._verbose):
            qf.smart_print(self._qc)

    def do_quite_step2(self):
        self._qc = qf.QuantumComputer(self._nqb)
        self._qc.apply_circuit(self._Uqite)
        btot = self.build_b2()
        A = qf.QuantumOperator()

        if(self._sparseSb):
            sp_idxs, S, btot = self.build_sparse_S_b2(btot)
        else:
            S = self.build_S2()

        x = lstsq(S, btot)[0]
        x = np.real(x)

        if(self._sparseSb):
            for I, spI in enumerate(sp_idxs):
                if np.abs(x[I]) > self._x_thresh:
                    A.add_term(-1.0j * self._db * x[I], self._sig.terms()[spI][1].terms()[0][1])

        else:
            for I, SigI in enumerate(self._sig.terms()):
                if np.abs(x[I]) > self._x_thresh:
                    A.add_term(-1.0j * self._db * x[I], SigI[1].terms()[0][1])

        if(self._verbose):
            print('\nbo:\n ')
            for val in self._bo:
                print('  ', val)
            print('\nbtot:\n ', btot)
            print('\n S:  \n')
            matprint(S)
            print('\n x:  \n')
            print(x)

        # np.savetxt('b_v2.dat', btot)
        # np.savetxt('s_v2.dat', S)

        eiA_kb, phase1 = trotterize(A, trotter_number=self._trotter_number)
        self._total_phase *= phase1
        self._Uqite.add_circuit(eiA_kb)

        self._qc.apply_circuit(eiA_kb)
        self._Ekb.append(np.real(self._qc.direct_op_exp_val(self._qb_ham)))

        if(self._verbose):
            qf.smart_print(self._qc)

    def evolve(self):
        self._Uqite.add_circuit(self._Uprep)
        print(' Beta        <Psi_b|H|Psi_b> ')
        print('---------------------------------------')
        print(' ', round(0.00, 3), '       ', np.round(self._Ekb[0], 10))

        for kb in range(1, self._nbeta):
            self.do_quite_step()
            print(' ', round(kb*self._db, 3), '       ', np.round(self._Ekb[kb], 10))

        self._Egs = self._Ekb[-1]

    def evolve2(self):
        self._Uqite.add_circuit(self._Uprep)
        print(' Beta        <Psi_b|H|Psi_b> ')
        print('---------------------------------------')
        print(' ', round(0.00, 3), '       ', np.round(self._Ekb[0], 10))

        for kb in range(1, self._nbeta):
            self.do_quite_step2()
            print(' ', round(kb*self._db, 3), '       ', np.round(self._Ekb[kb], 10))

        self._Egs = self._Ekb[-1]

    def print_expansion_ops(self):
        print('\nQITE expansion operators:')
        print('-------------------------')
        for I, SigI in enumerate(self._Sig):
            print('\nI: ', I)
            qf.smart_print(SigI)
