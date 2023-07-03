"""
Algorithms to do excited states based on expansions.  (E.g. q-sc-EOM)
"""
import qforte
import numpy as np


def q_sc_eom(H, U_ref, U_manifold):
    """
    Quantum, self-consistent equation-of-motion method from Asthana et. al.

    H is the JW-transformed Hamiltonian.
    U_ref is the VQE ground state circuit (or some other state not to be included in the manifold)
    U_manifold is a list of unitaries to be enacted on |0> to generate U_vqe|i> for each |i>.
    """
    N_qb = H.num_qubits()
    myQC = qforte.Computer(N_qb)
    myQC.apply_circuit(U_ref)
    E0 = myQC.direct_op_exp_val(H).real  
    Ek, A = ritz_eigh(H, U_manifold, verbose = False)
    print("q-sc-EOM:")
    print("*"*34)
    print(f"State:          Energy (Eh)")
    print(f"    0{E0:35.16f}")
    for i in range(0, len(Ek)):
        print(f"{(i+1):5}{Ek[i]:35.16f}")
    return E0, Ek   

def ritz_eigh(H, U, verbose = True):
    """
    Obtains the ritz eigeinvalues of H in the space of {U|i>}

    H is a qubit operator
    U is a list of unitaries
    """
    M = qforte.build_effective_operator(H, U)
    Ek, A = np.linalg.eigh(M)
    E_pre_diag = np.diag(M)
    if verbose == True:
        print("Ritz Diagonalization:")
        print(f"State:          Energy (Eh)")
        for i in range(len(Ek)):
            print(f"{i:5}{E_pre_diag[i]:35.16}{Ek[i]:35.16}")
    return Ek, A
    