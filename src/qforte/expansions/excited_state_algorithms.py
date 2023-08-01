"""
Algorithms to do excited states based on expansions.  (E.g. q-sc-EOM)
"""
import qforte
import numpy as np


def q_sc_eom(n_qubit, H, U_ref, U_manifold, ops_to_compute = []):
    """
    Quantum, self-consistent equation-of-motion method from Asthana et. al.

    H is the JW-transformed Hamiltonian.
    U_ref is the VQE ground state circuit (or some other state not to be included in the manifold)
    U_manifold is a list of unitaries to be enacted on |0> to generate U_vqe|i> for each |i>.
    ops_to_compute is a list of JW-transformed operators
    We will convert all of them into numpy arrays in the basis of {U_ref, U_manifold_i|0>}.

    """
    
    myQC = qforte.Computer(n_qubit)
    myQC.apply_circuit(U_ref)
    E0 = myQC.direct_op_exp_val(H).real
    print(f"Ground state energy: {E0}")
    print(f"Doing Ritz diagonalization for excited states.")  
    Ek, A = ritz_eigh(n_qubit, H, U_manifold) 
    
    op_mats = []
    if len(ops_to_compute) > 0:
        #Add the reference state with coefficient 1.
        n_states = len(Ek) + 1
        A_plus_ref = np.zeros((n_states, n_states), dtype = "complex")
        A_plus_ref[0, 0] = 1.0
        A_plus_ref[1:,1:] = A
        all_Us = [U_ref] + U_manifold
        
        for op in ops_to_compute:
            op_vqe_basis = qforte.build_effective_symmetric_operator(n_qubit, op, all_Us)
            op_q_sc_eom_basis = A_plus_ref.T.conj()@op_vqe_basis@A_plus_ref
            op_mats.append(op_q_sc_eom_basis)
    
    return [E0, Ek] + op_mats

def ritz_eigh(n_qubit, H, U, ops_to_compute = []):
    """
    Obtains the ritz eigeinvalues of H in the space of {U|i>}

    H is a qubit operator
    U is a list of unitaries
    ops_to_compute is a list of JW-transformed operators
    We will convert all of them into numpy arrays in the basis of {U_i|0>}.
    """
    M = qforte.build_effective_symmetric_operator(n_qubit, H, U)
    
    Ek, A = np.linalg.eigh(M)
    
    
    print("Ritz Diagonalization:")
    print(f"State:  Post-Diagonalized Energy")
    for i, E in enumerate(Ek):
        print(f"{(i+1):5}{E:35.16f}")

    op_mats = []
    
    for op in ops_to_compute:
        op_vqe_basis = qforte.build_effective_symmetric_operator(n_qubit, op, U)
        op_ritz_basis = A.T.conj()@op_vqe_basis@A
        op_mats.append(op_ritz_basis)
    
    return [Ek, A] + op_mats
    
