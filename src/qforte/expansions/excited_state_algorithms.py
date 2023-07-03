"""
Algorithms to do excited states based on expansions.  (E.g. q-sc-EOM)
"""
import qforte
import numpy as np


def q_sc_eom(H, U_ansatz, ref, manifold):
    """
    Quantum, self-consistent equation-of-motion method from Asthana et. al.

    H is the JW-transformed Hamiltonian.
    U_ansatz is the unitary such that U|ref> is the approximate ground state of H.
    ref is the unitary to prepare a reference state.
    manifold is the manifold of excited determinants.
    This might work out oddly for multi-determinant references...

    Set this up in a more realistic way to count gates, deal with noise, etc.
    """
    N_qb = H.num_qubits()
    
    myQC = qforte.Computer(N_qb)
    myQC.apply_circuit(ref)
    myQC.apply_circuit(U_ansatz)
    E0 = myQC.direct_op_exp_val(H).real

    M = np.zeros((len(manifold), len(manifold)), dtype = "complex")
    for i in range(0, len(manifold)):
        myQC = qforte.Computer(N_qb)
        myQC.apply_circuit(manifold[i])
        myQC.apply_circuit(U_ansatz)
        myQC.apply_operator(H)
        hket = np.array(myQC.get_coeff_vec())
        for j in range(i, len(manifold)):
            myQC = qforte.Computer(N_qb)
            myQC.apply_circuit(manifold[j])
            myQC.apply_circuit(U_ansatz)
            bra = np.array(myQC.get_coeff_vec())
            val = bra.T.conj()@hket 
            M[i,j] = M[j,i] = val
        
    Ek, A = np.linalg.eigh(M)
    print("q-sc-EOM:")
    print("*"*34)
    print(f"State:          Energy (Eh)")
    print(f"    0{E0:35.16f}")
    for i in range(0, len(Ek)):
        print(f"{(i+1):5}{Ek[i]:35.16f}")

    return E0, Ek
    
                    

