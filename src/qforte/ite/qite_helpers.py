"""
qite_helpers.py
=================================================
A class containing helper functions for quantum
imaginary time evolution.
"""
import qforte
from qforte.utils.transforms import (circuit_to_organizer,
                                    organizer_to_circuit,
                                    join_organizers,
                                    get_jw_organizer)

from qforte.utils.trotterization import trotterize

import numpy as np
from scipy.linalg import lstsq

class QITE(object):
    """
    A class that executes the quantum imaginary time evolution (QITE) algorithm.
    The ground state wave function is given by an infinite imaginary time step
    The basic QITE scheme is to approximate teh imaginary time operator with
    a unitary operator built as e^-iA where A is hermetian. A is an expansion
    over some set of Pauli operators (sigI), the choice of the expansion ultimately
    determines the classical cost of the method. The expansion coeficients at
    each time step (db) iteration (k) are found by solving a linear system Sx = b,
    where S_IJ = <Psi_k|SigI SigJ|Psi_k>,
    and b_I = (-i/sqrt(c)) * <Psi_k|H SigI|Psi_k>.
    Here, H is the Hamiltonian
    and c = 1 - 2*db*<Psi_k|H|Psi_k>
    is the normalization constat (approximated at 1st order).

    See https://arxiv.org/abs/1901.07653 for details (particulary the SI).

    Attributes
    ----------
    _ref : list
        The set of 1s and 0s indicating the initial quantum state.

    _nqubits : int
        The number of qubits the calculation empolys.

    _qb_operator : QuantumOperator
        The operator to be measured (usually the Hamiltonain), mapped to a
        qubit representation.

    _sq_operator : list
        The operator to be measured (usually the Hamiltonain), in a second
        quantized representaiton.

    _beta : float
        The total imaginary evolution time.

    _db : float
        The imaginary time step.

    _nbeta : int
        Then number of iterations (time-steps) the algorithm will use, specifically
        nbeta = beta/db + 1.

    _verbose : bool
        Whether or not to print additional details like the b vectors and the
        quantum computer state at each evolution time beta_k = k*db.

    _sparseSB : bool
        Whether or not to build a S and bl using only non-zero elements.

    _expansion_type : string
        The expansion basis for A.
        Can any one of the follwing,
            (1) 'sqGSD' : a basis of generalized second-quantized singles and
            doubels operators (still only for singlet states).
            (2) 'qbGSD' : same as above but does not cancel imaginary terms
            when converting to a pauli operator representaion.
            (3) 'complete_qubit' : all possible configurations of pauli operators,
            will result in a basis of size 4^nqubits.

    _state_prep : string
        How to use the reference to construct the initial state preperation
        circuit. Currently only supports 'single_reference'.

    _trott_number : int
        The Trotter number for the calculation
        (exact in the infinte limit).

    _fast : bool
        Whether or not to use a faster version of the algorithm that bypasses
        measurment (unphysical for quantum computer).

    _total_phase : float
        The accumulated total phase for the calculation, ultimately inconcequential
        for QITE, but relevant for QLanczos.

    _Uqite : QuantumCircuit
        The quantum circuit object which defines the QITE state at evolution time
        beta_k = k * nb. Circuit grows in depth with _nbeta and the size of the A
        expansion.

    _Ekb : list
        A list containing the QITE energy all evolution times kb = beta_k.
        _Ekb[0] is the SCF energy.

    _Nl : int
        Number of term in the qubit Hamiltonian.

    _NI : int
        Number of terms in the A expansion.

    _H : (_Nl X 1) numpy object array
        An array of single-term quantum operators, needed for building _HSig and
        calculating c.

    _Sig : (_NI X 1) numpy object array
        An array of single-term quantum operators representing the A exapnsion
        basis.

    _HSig : (_Nl X _NI) numpy object array
        An array of single-term quantum operators representing given by the
        products (Hl * SigI) of qubit hamiltoanin terms (Hl) and the A expansion basis
        terms (SigI). Used to construct bl.

    _SigSig : (_NI X _NI) numpy object array
        An array of single-term quantum operators representing given by all the
        products (SigI * SigJ) in _Sig. Used to construct S

    _n_blI_measurements : int
        (Not yet implemented)

    _n_SIJ_measurements : int
        (Not yet implemented)

    Methods
    -------
    to_base4()
        Converts an index to a base 4 string representaion.

    pauli_idx_str()
        Takes a base 4 string and ensures it is preceded by the
        appropriate number of '0's given the number of qubits
        in the calculation.

    build_Uprep()
        Builds the inital state preparation circuit. Usually the Hartree
        Fock state.

     build_expansion_pool()
        Builds the expansion for A. Populates the objects _H, _Sig, _HSig,
        and _SigSig.

    build_S()
        Measures the elements S_IJ and populates the S matrix.

    build_sparse_S_b()
        Optionally constructs S matrix and b vector using only non-zero values of
        btot.

    build_bl()
        Builds a vector bl corresponding to a particular pauli Hamiltonian
        term Hl.

    do_quite_step()
        Executes QITE for a small time step db. This solves Sx = b and updates
        _Uqite.

    evolve()
        Repeatedly calls do_qite_step until total evolution time _beta is reached.

    """
    def __init__(self, ref, qb_operator, sq_operator, beta, db,
                 verbose = False,
                 expansion_type = 'sqGSD',
                 state_prep = 'single_reference',
                 trotter_number = 1,
                 fast = True,
                 sparseSb = True):

        self._ref = ref
        self._nqubits = len(ref)
        self._qb_operator = qb_operator
        self._sq_operator = sq_operator
        self._beta = beta
        self._db = db
        self._nbeta = int(beta/db)+1
        self._verbose = verbose
        self._sparseSb = sparseSb
        self._expansion_type = expansion_type
        self._state_prep = state_prep
        self._trotter_number = trotter_number
        self._fast = fast
        self._total_phase = 1.0 + 0.0j
        self._Uqite = qforte.QuantumCircuit()

        self.build_Uprep()
        self.build_expansion_pool(combine=False)
        print('\n Expansion pool built: \n')

        qc_ref = qforte.QuantumComputer(self._nqubits)
        qc_ref.apply_circuit(self._Uprep)
        self._Ekb = [np.real(qc_ref.direct_op_exp_val(self._qb_operator))]

    def to_base4(self, I):
        convert_str = "0123456789"
        if I < 4:
            return convert_str[I]
        else:
            return self.to_base4(I//4) + convert_str[I%4]

    def pauli_idx_str(self, I_str):
        res = ''
        for i in range(self._nqubits-len(I_str)):
            res += '0'
        return res + I_str


    def build_Uprep(self): # TODO: put in more general file
        self._Uprep = qforte.QuantumCircuit()
        if self._state_prep == 'single_reference':
            for j in range(len(self._ref)):
                if self._ref[j] == 1:
                    self._Uprep.add_gate(qforte.make_gate('X', j, j))
        else:
            raise ValueError("Only 'single_reference' supported as state preparation type")

    def build_expansion_pool(self, combine=False):
        self._expansion_ops = []

        if(self._expansion_type == 'sqGSD'):
            SigI_org = []
            op_organizer = get_jw_organizer(self._sq_operator, combine=True)
            for l in range(len(op_organizer)):
                op_organizer[l][0] = 1.0 + 0.0j
                SigI_org.append([op_organizer[l]])

        elif(self._expansion_type == 'qbGSD'):
            op_organizer = get_jw_organizer(self._sq_operator, combine=False)
            uniqe_org = []
            SigI_org = []
            for term in op_organizer:
                for coeff, word in term:
                    uniqe_term = [1.0, word]
                    if(uniqe_term not in uniqe_org):
                        uniqe_org.append(uniqe_term)
                        SigI_org.append([uniqe_term])

        elif(self._expansion_type == 'complete_qubit'):
            if (self._nqubits > 6):
                raise ValueError('Using complete qubits expansion will reslut in a very large number of terms!')

            SigI_org = []
            paulis = ['I', 'X', 'Y', 'Z']
            for I in range(4**(self._nqubits)):
                paulistr = self.pauli_idx_str(self.to_base4(I))
                AI = []
                for k, gate_id in enumerate(paulistr):
                    if(gate_id != '0'):
                        AI.append( ( paulis[int(gate_id)], k ) )

                SigI_org.append([[1.0, AI]])

        else:
            raise ValueError('Invalid expansion type specified.')

        H_org = circuit_to_organizer(self._qb_operator)
        self._Nl = len(self._qb_operator.terms())
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

    def build_S(self):
        S = np.empty((self._NI,self._NI), dtype=complex)
        qc = qforte.QuantumComputer(self._nqubits)
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
        qc = qforte.QuantumComputer(self._nqubits)
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
        qc = qforte.QuantumComputer(self._nqubits)
        qc.apply_circuit(self._Uqite)
        for I in range(self._NI):
            bl[I] = bo * qc.direct_op_exp_val(self._HSig[l][I])

        return np.real(bl)


    def do_quite_step(self):
        qc = qforte.QuantumComputer(self._nqubits)
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
        self._Ekb.append(np.real(qc.direct_op_exp_val(self._qb_operator)))

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

    def print_expansion_ops(self):
        print('\nQITE expansion operators:')
        print('-------------------------')
        for I, SigI in enumerate(self._Sig):
            print('\nI: ', I)
            qforte.smart_print(SigI)
