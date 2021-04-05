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

from qforte.maths.eigsolve import canonical_geig_solve

### Throughout this file, we'll refer to DOI 10.1038/s41567-019-0704-4 as Motta.

class QITE(Algorithm):
    """
    Attributes
    ----------

    _b_thresh :
    _beta : float
    _db : float
    _do_lanczos : bool
    _Ekb : list of float
        The list of energies for each step of the algorithm.
    _expansion_type: {'complete_qubit', 'cqoy', 'SD', 'GSD', 'SDT', SDTQ', 'SDTQP', 'SDTQPH', 'test'}
        The family of operators that each evolution operator will be built of.
    _lanczos_gap : int
    _nbeta: int
        How many QITE steps should be taken?
    _NI : int
        The number of operators in _sig.
    _sig : QuantumOpPool
        The basis of operators allowed in a unitary evolution step.
    _sparseSb : bool
    _sq_ham : SqOperator
        The second-quantized, fermionic Hamiltonian
    _total_phase : complex
    _Uqite: QuantumCircuit
    _x_thresh : float
    """
    def run(self,
            beta=1.0,
            db=0.2,
            expansion_type='SD',
            sparseSb=True,
            b_thresh=1.0e-6,
            x_thresh=1.0e-10,
            do_lanczos=False,
            lanczos_gap=2):

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

        self._do_lanczos = do_lanczos
        self._lanczos_gap = lanczos_gap

        qc_ref = qf.QuantumComputer(self._nqb)
        qc_ref.apply_circuit(self._Uprep)
        self._Ekb = [np.real(qc_ref.direct_op_exp_val(self._qb_ham))]

        # Print options banner (should done for all algorithms).
        self.print_options_banner()

        # Build expansion pool.
        self.build_expansion_pool()

        # Do the imaginary time evolution.
        self.evolve()

        if (self._do_lanczos):
            self.do_qlanczos()

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
        if(not self._fast):
            print('Measurement variance thresh:             ',  0.01)

        # Specific QITE options.
        print('Total imaginary evolution time (beta):   ',  self._beta)
        print('Imaginary time step (db):                ',  self._db)
        print('Expansion type:                          ',  self._expansion_type)
        print('x value threshold:                       ',  self._x_thresh)
        print('Use sparse tensors to solve Sx = b:      ',  str(self._sparseSb))
        if(self._sparseSb):
            print('b value threshold:                       ',  str(self._b_thresh))
        print('\n')
        print('Do Quantum Lanczos                       ',  str(self._do_lanczos))
        if(self._do_lanczos):
            print('Lanczos gap size                         ',  self._lanczos_gap)

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
                raise ValueError('Using complete qubits expansion will result in a very large number of terms!')
            self._sig.fill_pool("complete_qubit", self._ref)

        elif(self._expansion_type == 'cqoy'):
            self._sig.fill_pool("cqoy", self._ref)

        elif(self._expansion_type in {'SD', 'GSD', 'SDT', 'SDTQ', 'SDTQP', 'SDTQPH'}):
            P = qf.SQOpPool()
            P.set_orb_spaces(self._ref)
            P.fill_pool(self._expansion_type)
            sig_temp = P.get_quantum_operator("commuting_grp_lex", False)

            # Filter the generated operators, so that only those with an odd number of Y gates are allowed.
            # See section "Real Hamiltonians and states" in the SI of Motta for theoretical justification.
            # Briefly, this method solves Ax=b, but all b elements with an odd number of Y gates are imaginary and
            # thus vanish. This method will not be correct for non-real Hamiltonians or states.
            for alph, rho in sig_temp.terms():
                nygates = 0
                temp_rho = qf.QuantumCircuit()
                for gate in rho.gates():
                    temp_rho.add_gate(qf.make_gate(gate.gate_id(), gate.target(), gate.control()))
                    if (gate.gate_id() == "Y"):
                        nygates += 1

                if (nygates % 2 == 1):
                    rho_op = qf.QuantumOperator()
                    rho_op.add_term(1.0, temp_rho)
                    self._sig.add_term(1.0, rho_op)

        elif(self._expansion_type == 'test'):
            self._sig.fill_pool("test", self._ref)

        else:
            raise ValueError('Invalid expansion type specified.')

        self._NI = len(self._sig.terms())


    def build_S(self):
        """
        Construct the matrix S (eq. 5a) of Motta.
        """
        Idim = self._NI

        S = np.zeros((Idim, Idim), dtype=complex)

        Ipsi_qc = qf.QuantumComputer(self._nqb)
        Ipsi_qc.set_coeff_vec(copy.deepcopy(self._qc.get_coeff_vec()))
        # CI[I][J] = (σ_I Ψ)_J
        CI = np.zeros(shape=(Idim, int(2**self._nqb)), dtype=complex)

        for i in range(Idim):
            S[i][i] = 1.0 # With Pauli strings, this is always the inner product
            Ipsi_qc.apply_operator(self._sig.terms()[i][1])
            CI[i,:] = copy.deepcopy(Ipsi_qc.get_coeff_vec())
            for j in range(i):
                S[i][j] = S[j][i] = np.vdot(CI[i,:], CI[j,:])
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

        for i in range(Idim):
            S[i][i] = 1.0 # With Pauli strings, this is always the inner product
            Ii = idx_sparse[i]
            Ipsi_qc.apply_operator(self._sig.terms()[Ii][1])
            CI[i,:] = copy.deepcopy(Ipsi_qc.get_coeff_vec())
            for j in range(i):
                S[i][j] = S[j][i] = np.vdot(CI[i,:], CI[j,:])
            Ipsi_qc.set_coeff_vec(copy.deepcopy(self._qc.get_coeff_vec()))

        return idx_sparse, np.real(S), np.real(b_sparse)

    def build_b(self):
        """
        Construct the vector b (eq. 5b) of Motta, with h[m] the full Hamiltonian.
        """

        b  = np.zeros(self._NI, dtype=complex)

        denom = np.sqrt(1 - 2*self._db*self._Ekb[-1])
        prefactor = -1.0j / denom

        self._n_pauli_trm_measures += self._Nl * self._NI

        Hpsi_qc = qf.QuantumComputer(self._nqb)
        Hpsi_qc.set_coeff_vec(copy.deepcopy(self._qc.get_coeff_vec()))
        Hpsi_qc.apply_operator(self._qb_ham)
        C_Hpsi_qc = copy.deepcopy(Hpsi_qc.get_coeff_vec())

        for I, (op_coefficient, operator) in enumerate(self._sig.terms()):
            Hpsi_qc.apply_operator(operator)
            exp_val = np.vdot(self._qc.get_coeff_vec(), Hpsi_qc.get_coeff_vec())
            b[I] = prefactor * op_coefficient * exp_val
            Hpsi_qc.set_coeff_vec(C_Hpsi_qc)

        return np.real(b)

    def do_qite_step(self):

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

        if(self._do_lanczos):
            self._lanczos_vecs = []
            self._Hlanczos_vecs = []

            self._lanczos_vecs.append(copy.deepcopy(self._qc.get_coeff_vec()))

            qcSig_temp = qf.QuantumComputer(self._nqb)
            qcSig_temp.set_coeff_vec(copy.deepcopy(self._qc.get_coeff_vec()))
            qcSig_temp.apply_operator(self._qb_ham)
            self._Hlanczos_vecs.append(copy.deepcopy(qcSig_temp.get_coeff_vec()))


        print(f"{'beta':>7}{'E(beta)':>18}{'N(params)':>14}{'N(CNOT)':>18}{'N(measure)':>20}")
        print('-------------------------------------------------------------------------------')
        print(f' {0.0:7.3f}    {self._Ekb[0]:+15.9f}    {self._n_classical_params:8}        {self._n_cnot:10}        {self._n_pauli_trm_measures:12}')

        if (self._print_summary_file):
            f = open("summary.dat", "w+", buffering=1)
            f.write(f"#{'beta':>7}{'E(beta)':>18}{'N(params)':>14}{'N(CNOT)':>18}{'N(measure)':>20}\n")
            f.write('#-------------------------------------------------------------------------------\n')
            f.write(f'  {0.0:7.3f}    {self._Ekb[0]:+15.9f}    {self._n_classical_params:8}        {self._n_cnot:10}        {self._n_pauli_trm_measures:12}\n')

        for kb in range(1, self._nbeta):
            self.do_qite_step()
            if(self._do_lanczos):
                if(kb % self._lanczos_gap == 0):
                    self._lanczos_vecs.append(copy.deepcopy(self._qc.get_coeff_vec()))

                    qcSig_temp = qf.QuantumComputer(self._nqb)
                    qcSig_temp.set_coeff_vec(copy.deepcopy(self._qc.get_coeff_vec()))
                    qcSig_temp.apply_operator(self._qb_ham)
                    self._Hlanczos_vecs.append(copy.deepcopy(qcSig_temp.get_coeff_vec()))

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


    def do_qlanczos(self):
        """"""
        n_lanczos_vecs = len(self._lanczos_vecs)
        h_mat = np.zeros((n_lanczos_vecs,n_lanczos_vecs), dtype=complex)
        s_mat = np.zeros((n_lanczos_vecs,n_lanczos_vecs), dtype=complex)

        print('\n\n-----------------------------------------------------')
        print('         Quantum Imaginary Time Lanczos   ')
        print('-----------------------------------------------------\n\n')


        print(f"{'Beta':>7}{'k(S)':>7}{'E(Npar)':>19}")
        print('-------------------------------------------------------------------------------')

        if (self._print_summary_file):
            f2 = open("lanczos_summary.dat", "w+", buffering=1)
            f2.write(f"#{'Beta':>7}{'k(S)':>7}{'E(Npar)':>19}\n")
            f2.write('#-------------------------------------------------------------------------------\n')

        for m in range(n_lanczos_vecs):
            for n in range(m+1):
                h_mat[m][n] = np.vdot(self._lanczos_vecs[m], self._Hlanczos_vecs[n])
                h_mat[n][m] = np.conj(h_mat[m][n])
                s_mat[m][n] = np.vdot(self._lanczos_vecs[m], self._lanczos_vecs[n])
                s_mat[n][m] = np.conj(s_mat[m][n])

            k = m+1
            evals, evecs = canonical_geig_solve(s_mat[0:k, 0:k],
                               h_mat[0:k, 0:k],
                               print_mats=False,
                               sort_ret_vals=True)

            scond = np.linalg.cond(s_mat[0:k, 0:k])

            print(f'{m * self._lanczos_gap * self._db:7.3f} {scond:7.2e}    {np.real(evals[0]):+15.9f} ')
            if (self._print_summary_file):
                f2.write(f'{m * self._lanczos_gap * self._db:7.3f} {scond:7.2e}    {np.real(evals[0]):+15.9f} \n')

        if (self._print_summary_file):
            f2.close()

        self._Egs_lanczos = evals[0]
