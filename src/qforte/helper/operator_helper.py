import qforte
from openfermion.ops import QubitOperator
import numpy as np

def build_from_openfermion(OF_qubitops):

    """
    build_from_openfermion is a function that build a QuantumOperator instance in
    qforte from a openfermion QubitOperator instance

    :param OF_qubitops: the QubitOperator instance from openfermion.
    """

    qforte_ops = qforte.QuantumOperator()
    #for term, coeff in sorted(OF_qubitops.terms.items()):
    for term, coeff in OF_qubitops.terms.items():
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
