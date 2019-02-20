import qforte
from openfermion.ops import QubitOperator
import numpy as np

def build_from_openferm(OF_qubitops):
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
    
test_operator = QubitOperator('X2 Y1', 0.0-0.25j)
test_operator += QubitOperator('Y2 Y1', 0.25)
test_operator += QubitOperator('X2 X1', 0.25)
test_operator += QubitOperator('Y2 X1', 0.0+0.25j)
print(test_operator)

qforte_operator = build_from_openferm(test_operator)
#print('\n'.join(qforte_operator.str()))

