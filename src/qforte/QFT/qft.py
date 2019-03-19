import qforte
import numpy

def qft_circuit(State):

    if(isinstance(qforte.QuantumComputer)):
        continue
    else
        return -1

    # 1. Build R gates as a list

    # 2. Build qft circuit: 

    return qft_circ

def reverse_circuit(circ):

    return rev_circ

def qft_trans(state):

    circ = qft_circuit(state)
    state.apply_circuit(circ)

    return state

def qft_rev_trans(state):
    
    circ = qft_circuit(state)
    rev_circ = reverse_circuit(circ)
    state.apply_circuit(rev_circ)

    return state          
