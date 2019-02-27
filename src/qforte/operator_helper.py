import qforte
from openfermion.ops import QubitOperator
import numpy as np

def build_from_openferm(OF_qubitops):

    #Build QuantumOperator from openfermion QubitOperator
    qforte_ops = qforte.QuantumOperator()
    for term, coeff in sorted(OF_qubitops.terms.items()):
        circ_term = qforte.QuantumCircuit()
        #Exclude zero terms
        if np.isclose(coeff, 0.0):
            continue
        for factor in term:
            index, action = factor
            # Read the string name for actions(gates)
            action_string = OF_qubitops.action_strings[OF_qubitops.actions.index(action)]

            #Make qforte gates and add to circuit
            gate_this = qforte.make_gate(action_string, index, index)
            circ_term.add_gate(gate_this)

        #Add this term to operator
        qforte_ops.add_term(coeff, circ_term)

    return qforte_ops

def smart_print(Inputobj):
    
    #Assert the class and print smartly
    if isinstance(Inputobj, qforte.QuantumOperator):
        print('\n Quantum operator:')
        ops_term = Inputobj.terms()
        for term in ops_term:
            print('\n')
            print(term[0])
            print('\n'.join(term[1].str()))

    if isinstance(Inputobj, qforte.QuantumCircuit):
        print('\n Quantum circuit:')
        print('\n'.join(Inputobj.str()))

    if isinstance(Inputobj, qforte.QubitOperator):
        print('\n Openfermion Qubit operator:')
        print(Inputobj)
