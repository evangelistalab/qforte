import openfermion
from openfermion.ops import FermionOperator
from openfermion.transforms import jordan_wigner, get_fermion_operator
from openfermion.utils import hermitian_conjugated, commutator, normal_ordered

import numpy as np 
import scipy

import qforte
# from qforte.utils.trotterization import *
from qforte.utils.exponentiate import *

from openfermionpsi4 import *


def runPsi4(geometry, **kwargs):
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
                        run_ccsd=kwargs.get('run_ccsd', 1),
                        run_fci=kwargs.get('run_fci', 1))
    return molecule


class ADAPT_VQE:
    """
    Attributes
    ----------
    molecule: instance of openfermion.hamiltonians.MolecularData class.

    jw_ops: dict
        keys: A_i, qforte.QuantumOperator instances
        values: commutator [H, A_i], qforte.QuantumOperator instances
    """
    def __init__(self, molecule_instance):
        self.molecule = molecule_instance
        self.jw_ops = []
        self.jw_commutators = []

        self.occ_idx_alpha = [
            2*i for i in range(molecule_instance.get_n_alpha_electrons())]
        self.occ_idx_beta = [
            2*i+1 for i in range(molecule_instance.get_n_beta_electrons())]
        self.vir_idx_alpha = [2*a for a in range(
            self.molecule.get_n_alpha_electrons(), molecule_instance.n_orbitals)]
        self.vir_idx_beta = [2*a+1 for a in range(
            self.molecule.get_n_beta_electrons(), molecule_instance.n_orbitals)]
        
        self.ref_state = self.ref_state()
        # self.param_list = []
        self.ansatz_ops = []
        self.ansatz_op_idx = []


    def build_operator_pool(self, docc_indices=None, active_orb_indices=None):
        molecular_hamiltonian = self.molecule.get_molecular_hamiltonian(
            occupied_indices=Nodocc_indices, active_indices=active_orb_indices)
        h_fermion = normal_ordered(get_fermion_operator(molecular_hamiltonian))
        h_qubit = jordan_wigner(h_fermion)

        # Singles
        def add_singles(occ_idx, vir_idx):
            for i in occ_idx_alpha:
                for a in vir_idx_alpha:
                    single = FermionOperator(((a, 1), (i, 0)))
                    # 1. build Fermion anti-Hermitian operator
                    single -= hermitian_conjugated(single)
                    # 2. JW transformation to qubit operator
                    jw_single = jordan_wigner(single)
                    h_single_commutator = commutator(h_qubit, jw_single)
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
                    h_double_commutator = commutator(h_qubit, jw_double)

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

        occ_ab_pairs = list(product(self.occ_idx_alpha, occ_idx_beta))
        vir_ab_pairs = list(product(self.vir_idx_alpha, vir_idx_beta))
        add_doubles(occ_ab_pairs, vir_ab_pairs)


    def ref_state(self):
        """ Return a qforte.QuantumComputer instance which represents the Hartree-Fock state of the input molecule
        """
        hf_qc = qforte.QuantumComputer(self.molecule.n_qubits)
        hf_cir = qforte.QuantumCircuit()

        for i_a in occ_idx_alpha:
            X_ia = qforte.make_gate('X', i_a, i_a)
            hf_cir.add_gate(X_ia)
        
        for i_b in occ_idx_beta:
            X_ib = qforte.make_gate('X', i_b, i_b)
            hf_cir.add_gate(X_ib)
        
        hf_qc.apply_circuit(hf_cir)
        return hf_qc

    def get_ansatz_circuit(self, param_list):
        """ Return a qforte.QuantumCircuit object parametrized by input param_list
        Parameters
        ----------
        param_list: list
            list of parameters (rotation angles) used for preparing circuit applied on the reference state 
            (NOT the circuit to prepare reference state)
        """
        param_circuit = qforte.QuantumCircuit()
        for i in range(len(param_list)):
            param_i = param_list[i]
            idx, param = param_list[i]
            op = self.jw_ops[i]
            # exp_op is a circuit object
            exp_op = qforte.QuantumCircuit()

            for coeff, term in op.terms():
                factor = coeff*param_i
                #'exponentiate_single_term' function returns a tuple (exponential(Cir), 1.0)
                exp_term = exponentiate_single_term(factor, term)[0]
                exp_op.add_circuit(exp_term)

            param_circuit.add_circuit(exp_op)
        return param_circuit

    def compute_gradient(self, param_list, use_analytic_grad=True, numerical_grad_precision=1e-3):
        """ Return a list of energy gradients for all operators in jw_ops list

        Parameters
        ----------
        phi_i: qforte.QuantumComputer instance
            quantum state / trial wavefunction at step i

        Returns
        -------
        gradients: list
            energy gradient w.r.t. param_i
        """
        gradients = []
        if param_list == None:
            phi_i = self.ref_state
        else: 
            param_circuit = self.get_ansatz_circuit(param_list)
            phi_i = self.ref_state.apply_circuit(param_circuit)

        if use_analytic_grad:
            for op in self.jw_ops:
                commutator = self.jw_commutators[self.jw_ops.index(op)]
                term_sum = 0.0
                n_terms = len(commutator.terms())

                for i in range(n_terms):
                    term_sum += commutator.terms()[i][0] * \
                        phi_i.perfect_measure_circuit(commutator.terms()[i][1])
                
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
            pass


        return gradients


    # def compute_expectation(self, param_list, ref_state):
    #     """ Return energy 
    #     """
    #     param_circuit = self.get_ansatz_circuit(param_list)
    #     energy = ref_state.perfect_measure_circuit(param_circuit)
    #     return energy

    def compute_expectation(self, param_list):
        """ Return <phi_i|H|phi_i>
        """
        param_circuit = self.get_ansatz_circuit(param_list)
        expectation = self.ref_state.perfect_measure_circuit(param_circuit)
        return expectation

    def get_variational_energy(input_params):
        result = scipy.optimize.minimize(compute_expectation, input_params, method= ‘BFGS’,\
                jac=Trotter_Gradient, \
                options={‘gtol’: float(theta_tightness), ‘disp’: False}, callback=Callback)

    # lambda H, state: 
    def iterating_cycle(self, max_cycle = 100, gradient_norm_threshold = 1e-3):
        # 3. initialize |HF> state, select op1 from jw_ops list
        phi0 = self.ref_state

        params = []
        # for i in range(len(jw_ops_list)):
        #     gradients_i = 
        gradients_0 = self.compute_gradient(phi0)
        idx_max_grad = gradients_0.index(max(gradients_0))
        A_1 = jw_ops_list[idx_max_grad]
        param_1 = 0.0
        params.append((idx_max_grad, param_1))
        
       

        cir_trot, phase = trotterize(operator)
        # 4. prepare state |phi1> = exp(op1*param1)|HF>
        # 5. gradients measure   
        #    Input: list of op;     Output: list of gradient values
        # 6. select op2 with largest gradient, prepare state |phi2> = exp(op2*param2)|phi1>
        #  minimize E = <phi2|H|phi2>
        


        pass
        
        






# def vqe_minimizer(initial_params):
#     """ Return an updated qforte QuantumComputer object
    
#     Parameters
#     ----------
#     initial_params: list
#         List of parameters which define the ansatz
#     ansatz: string
#         'ucc', 

#     Returns
#     -------
#     updated_quantum_computer: qforte.QuantumComputer object
#         QuantumComputer object which parametrized using new list of parameters
#     """


#     return updated_quantum_computer
#     pass
