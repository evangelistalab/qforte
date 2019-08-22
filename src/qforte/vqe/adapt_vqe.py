import openfermion
from openfermion.ops import FermionOperator
from openfermion.transforms import jordan_wigner, get_fermion_operator
from openfermion.utils import hermitian_conjugated, commutator, normal_ordered
from openfermionpsi4 import *
import numpy as np 
import scipy
import qforte
from qforte.utils.exponentiate import exponentiate_single_term

def runPsi4(geometry, kwargs):
    """ Returns an updated MolecularData object

    Parameters
    ----------
    geometry: list
        A list of tuples giving the coordinates of each atom.
        An example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))]
        Distances in angstrom. Use atomic symbols to specify atoms.
    kwargs: dict
        A dictionary to set up psi4 calculation
        keys: basis, multiplicity, charge, description, run_scf, run_mp2, run_cisd, run_ccsd, run_fci

    """
    basis = kwargs.get('basis', 'sto-3g')
    multiplicity = kwargs.get('multiplicity', 1)
    charge = kwargs.get('charge', 0)
    description = kwargs.get('description', '')

    molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity, charge, description)

    molecule = run_psi4(molecule,
                        run_scf=kwargs.get('run_scf', 1),
                        run_mp2=kwargs.get('run_mp2', 0),
                        run_cisd=kwargs.get('run_cisd', 0),
                        run_ccsd=kwargs.get('run_ccsd', 0),
                        run_fci=kwargs.get('run_fci', 0))
    return molecule


class ADAPT_VQE:
    """ ADAPT_VQE class is initiaized with an instance of openfermion.hamiltonians.MolecularData

    Attributes
    ----------
    molecule: instance of openfermion.hamiltonians.MolecularData class.
    jw_ops: list of qforte.QuantumOperator instances (A_i)
            obtained by jordan-wigner encoding all fermionic singles and doubles excitation operators.
    jw_commutators: list of qforte.QuantumOperator instances (commutator [H, A_i])

    occ_idx_alpha: list of indices of qubits encoding occupied alpha spin orbitals (even indices)
    occ_idx_beta: list of indices of qubits encoding occupied beta spin orbitals (odd indices)
    vir_idx_alpha: list of indices of qubits encoding virtual alpha spin orbitals (even indices)
    vir_idx_beta: list of indices of qubits encoding virtual beta spin orbitals (odd indices)

    ref_state: instance of qforte.QuantumComputer class
    ansatz_ops: list of qforte.QuantumOperator instances which prepare the wavefunction ansatz
    ansatz_op_idx: list of "jw_ops-indices" of operators in ansatz_ops list
    energy: list of variational optimized energies in all ADAPT cycles before convergence

    """
    def __init__(self, molecule_instance):
        self.molecule = molecule_instance
        self.jw_ops = []
        self.jw_commutators = []
        self.qubit_hamiltonian, self.h_qubit_OF = self.get_qubit_hamitonian()

        self.occ_idx_alpha = [
            2*i for i in range(molecule_instance.get_n_alpha_electrons())]
        self.occ_idx_beta = [
            2*i+1 for i in range(molecule_instance.get_n_beta_electrons())]
        self.vir_idx_alpha = [2*a for a in range(
            self.molecule.get_n_alpha_electrons(), molecule_instance.n_orbitals)]
        self.vir_idx_beta = [2*a+1 for a in range(
            self.molecule.get_n_beta_electrons(), molecule_instance.n_orbitals)]
        
        # self.ref_state = self.ref_state()
        self.ansatz_ops = []
        self.ansatz_op_idx = []
        self.energy = []

    # @property
    # def qubit_hamiltonian(self):
    #     return self._qubit_hamiltonian

    # @qubit_hamiltonian.setter
    def get_qubit_hamitonian(self, docc_indices=None, active_orb_indices=None):
        """ Return Hamiltonian operator as a qforte.QuantumOperator instance

        Parameters
        ----------
        docc_indices: list, optional
            list of spatial orbital indices indicating which orbitals should be considered doubly occupied.
        active_orb_indices: list, optional
            list of spatial orbital indices indicating which orbitals should be considered active.

        """
        # "molecular_hamiltonian": instance of the MolecularOperator class
        
        molecular_hamiltonian = self.molecule.get_molecular_hamiltonian(
            occupied_indices=docc_indices, active_indices=active_orb_indices)
        h_fermion = normal_ordered(get_fermion_operator(molecular_hamiltonian))
        h_qubit_OF = jordan_wigner(h_fermion)
        h_qubit_qf = qforte.build_from_openfermion(h_qubit_OF)
        return h_qubit_qf, h_qubit_OF 

    def build_operator_pool(self):
        """ Call this function to populate jw_ops list and jw_commutators list
        """
        # Singles
        def add_singles(occ_idx, vir_idx):
            for i in occ_idx:
                for a in vir_idx:
                    single = FermionOperator(((a, 1), (i, 0)))
                    # 1. build Fermion anti-Hermitian operator
                    single -= hermitian_conjugated(single)
                    # 2. JW transformation to qubit operator
                    jw_single = jordan_wigner(single)
                    h_single_commutator = commutator(self.h_qubit_OF, jw_single)
                    # 3. qforte.build_from_openfermion(OF_qubitop)
                    qf_jw_single = qforte.build_from_openfermion(jw_single)
                    qf_commutator = qforte.build_from_openfermion(h_single_commutator)

                    self.jw_ops.append(qf_jw_single)
                    self.jw_commutators.append(qf_commutator)

        add_singles(self.occ_idx_alpha, self.vir_idx_alpha)
        add_singles(self.occ_idx_beta, self.vir_idx_beta)
                
        # Doubles
        def add_doubles(occ_idx_pairs, vir_idx_pairs):
            for ji in occ_idx_pairs:
                for ba in vir_idx_pairs:
                    j, i = ji
                    b, a = ba

                    double = FermionOperator(F'{a}^ {b}^ {i} {j}')
                    double -= hermitian_conjugated(double)
                    jw_double = jordan_wigner(double)
                    h_double_commutator = commutator(self.h_qubit_OF, jw_double)

                    qf_jw_double = qforte.build_from_openfermion(jw_double)
                    qf_commutator = qforte.build_from_openfermion(
                    h_double_commutator)

                    self.jw_ops.append(qf_jw_double)
                    self.jw_commutators.append(qf_commutator)

        from itertools import combinations, product

        occ_a_pairs = list(combinations(self.occ_idx_alpha, 2))
        vir_a_pairs = list(combinations(self.vir_idx_alpha, 2))
        add_doubles(occ_a_pairs, vir_a_pairs)

        occ_b_pairs = list(combinations(self.occ_idx_beta, 2))
        vir_b_pairs = list(combinations(self.vir_idx_beta, 2))
        add_doubles(occ_b_pairs, vir_b_pairs)

        occ_ab_pairs = list(product(self.occ_idx_alpha, self.occ_idx_beta))
        vir_ab_pairs = list(product(self.vir_idx_alpha, self.vir_idx_beta))
        add_doubles(occ_ab_pairs, vir_ab_pairs)

    @property
    def ref_state(self):
        """ Return a qforte.QuantumComputer instance which represents the Hartree-Fock state of the input molecule
        """
        hf_qc = qforte.QuantumComputer(self.molecule.n_qubits)
        hf_cir = qforte.QuantumCircuit()

        for i_a in self.occ_idx_alpha:
            X_ia = qforte.make_gate('X', i_a, i_a)
            hf_cir.add_gate(X_ia)
        
        for i_b in self.occ_idx_beta:
            X_ib = qforte.make_gate('X', i_b, i_b)
            hf_cir.add_gate(X_ib)
        
        hf_qc.apply_circuit(hf_cir)
        return hf_qc

    def get_ansatz_circuit(self, param_list):
        """ Return a qforte.QuantumCircuit object parametrized by input param_list
        
        Parameters
        ----------
        param_list: list
            list of parameters [param_1,..., param_n]
            to prepare the circuit 'exp(param_n*A_n)...exp(param_1*A_1)'
        
        Returns
        -------
        param_circuit: instance of qforte.QuantumCircuit class
            the circuit to be applied on the reference state to get wavefunction ansatz. 
        
        """
        param_circuit = qforte.QuantumCircuit()
        for i in range(len(param_list)):
            param_i = param_list[i]
            op = self.ansatz_ops[i]
            # exp_op is a circuit object
            exp_op = qforte.QuantumCircuit()

            for coeff, term in op.terms():
                factor = coeff*param_i
                #'exponentiate_single_term' function returns a tuple (exponential(Cir), 1.0)
                exp_term = exponentiate_single_term(factor, term)[0]
                exp_op.add_circuit(exp_term)

            param_circuit.add_circuit(exp_op)
        return param_circuit

    def compute_gradient(self, param_list, use_analytic_grad=True, numerical_grad_precision=1e-3, gvec_abs = True):
        """ Return a list of energy gradients for all operators in jw_ops list

        Parameters
        ----------
        param_list: list
            list of parameters [param_1,..., param_n]
            to prepare the circuit 'exp(param_n*A_n)...exp(param_1*A_1)'
        use_analytic_grad: boolean

        Returns
        -------
        gradients: list
            list of energy gradients (Absolute Value) w.r.t. all param_i in the jw_ops list
        """
        gradients = []

        param_circuit = self.get_ansatz_circuit(param_list)
        current_wfn = self.ref_state
        current_wfn.apply_circuit(param_circuit)

        if use_analytic_grad:

            for i in range(len(self.jw_ops)):
                commutator = self.jw_commutators[i]
                term_sum = 0.0

                ##Use inner product as measurement
                # dot = current_wfn.direct_op_exp_val(commutator)
                # dot = np.real(dot)
                # gradients.append(dot)
                for term_i in commutator.terms():
                    term_sum += term_i[0] * current_wfn.perfect_measure_circuit(term_i[1])
                # n_terms = len(commutator.terms())
                # for i in range(n_terms):
                #     term_sum += commutator.terms()[i][0] * \
                #         current_wfn.perfect_measure_circuit(commutator.terms()[i][1])
                if gvec_abs:
                    term_sum = np.abs(np.real(term_sum))
                else: 
                    term_sum = np.real(term_sum)
                gradients.append(term_sum)
        else:
            # use numerical gradients 
            """ Numerical approach only computes gradients of parameters 
                corresponding to operators included in wavefunction ansatz.
            """
            # for i in range(len(param_list)):

            #     param_list1 = param_list.copy()
            #     param_list1[i] = param_list[i] + numerical_grad_precision
            #     E_i_plus = compute_energy(param_list1, self.ref_state)
            #     param_list2 = param_list.copy()
            #     param_list2[i] = param_list[i] - numerical_grad_precision
            #     E_i_minus = compute_energy(param_list2, self.ref_state)

            #     gradient_i = (E_i_plus - E_i_minus) /(2*numerical_grad_precision)
            #     gradients.append(gradient_i)
            print('Numerical gradient not implemented yet.')


        return gradients


    def compute_expectation(self, param_list):
        """ Return <wfn|H|wfn>

        Parameters
        ----------
        param_list: list
            list of parameters [param_1,..., param_n]
            to prepare the circuit 'exp(param_n*A_n)...exp(param_1*A_1)'

        Returns
        -------
        expectation: float
            <HF|exp(-param_1*A_1)...exp(-param_n*A_n)*H*exp(param_n*A_n)...exp(param_1*A_1)|HF>
        """
        param_circuit = self.get_ansatz_circuit(param_list)
        current_wfn = self.ref_state
        current_wfn.apply_circuit(param_circuit)

        expectation = 0.0
        for h_i in self.qubit_hamiltonian.terms():
            expectation += h_i[0] * current_wfn.perfect_measure_circuit(h_i[1])

        expectation = np.real(expectation)
        return expectation

    def variational_optimizer(self, input_params):
        """ Return scipy.optimize.OptimizeResult object

        Parameters
        ----------
        input_params: list
            list of paramaters which prepares the wavefunction ansatz

        Returns
        -------
        result: a scipy.optimize.OptimizeResult object with important attributes:
                - result.x: (ndarray) optimized parameters
                - result.success: (bool) whether or not the optimizer exited successfully
                - result.fun: (float) minimized expectation value (energy)

                - result.message: (string) description of the cause of the termination
        """
        print('        ---------------- Start VQE ----------------')
        print('        Optimizing parameters to minimize energy...')
        # jac = compute_gradient
        # result = scipy.optimize.minimize(self.compute_expectation, input_params, method='BFGS', options={'gtol': 1e-2})
        input_param_array = np.asarray(input_params)
        result = scipy.optimize.minimize(
            self.compute_expectation, input_param_array, method='BFGS')

        print('        '+str(len(input_params)) +
              ' parameters optimized after ' + str(result.nit) + ' iterations')
        print('        ---------------- Finish VQE ----------------')
        return result

    def iterating_cycle(self, max_cycle= 100, gradient_norm_threshold = 1e-2, print_details = True, restart = False):
        if not restart:
            print('\n    ================> Start ADAPT-VQE <================\n')
            params = []
        else:
            print('\n    ================> reStart ADAPT-VQE <================\n')
            print(F'    NOT support restart yet.')


        for i in range(1, max_cycle+1):
            print(F'    ====== ADAPT cycle {i} ======')
            gradients_i = self.compute_gradient(params)
            # print(F'    gradient_vector = {gradients_i}')
            if np.linalg.norm(gradients_i) < gradient_norm_threshold:
                print('\n    =========> Finish ADAPT-VQE Successfully! <=========')
                print(F'    Norm of gradient vector Converged! (norm less than {gradient_norm_threshold})\n')
                if print_details:
                    print('\n--- Details ---')
                    print('#op in op pool = ', len(self.jw_ops))
                    print('#commutator = ', len(self.jw_commutators))
                    print('#ansatzOp = ', len(self.ansatz_ops))
                    print('#ansatzOpIdx = ', len(self.ansatz_op_idx))
                    print('Energy from 1st ADAPT iteration: {} Hartree'.format(self.energy[0]))
                    print('Energy from final ADAPT iteration: {} Hartree'.format(self.energy[-1]))
                    print('\n--- reference ---')
                    print('psi4 HF  energy: {} Hartree.'.format(self.molecule.hf_energy))
                    print('psi4 FCI energy: {} Hartree.'.format(self.molecule.fci_energy))
                break
                
            else:
                idx_of_max_grad_i = gradients_i.index(max(gradients_i))
                self.ansatz_op_idx.append(idx_of_max_grad_i)
                print(F'    ansatz_op_idx = {self.ansatz_op_idx}')

                A_i = self.jw_ops[idx_of_max_grad_i]
                self.ansatz_ops.append(A_i)
                # print(F'    ansatz_ops = {self.ansatz_ops}')

                param_i = 0.005
                params = np.append(params, param_i)
                # print(F'    Current parameters = {params}')

                result = self.variational_optimizer(params)
                params = result.x
                # print(F'    Optimized parameters = {params}')
                energy_i = result.fun
                self.energy.append(energy_i)
                print(F'    Optimized Energy of ADAPT cycle {i}: {energy_i}\n')
        else: 
            print(F'Warning: Norm of gradients vector did not converge (less than {gradient_norm_threshold}) in {max_cycle} cycles')


    
