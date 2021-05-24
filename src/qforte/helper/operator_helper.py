import qforte
import numpy as np

def build_from_openfermion(OF_qubitops, time_evo_factor = 1.0):

    """
    builds a QuantumOperator instance in
    qforte from a openfermion QubitOperator instance

    :param OF_qubitops: (QubitOperator) the QubitOperator instance from openfermion.
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
            gate_this = qforte.gate(action_string, index, index)
            circ_term.add_gate(gate_this)

        #Add this term to operator
        qforte_ops.add_term(coeff*time_evo_factor, circ_term)

    return qforte_ops

def build_sqop_from_openfermion(OF_fermiops, time_evo_factor = 1.0):
    """Builds a list representing the normal ordered operator OF_fermiops that includes
    the indices of spin orbital anihilators and creators, and the respective
    coefficient. The OF_fermiops must be alredy normal ordered.

    Arguments
    ---------

    OF_fermiops : QubitOperator
        the QubitOperator instance from openfermion.

    Returns
    -------

    sq_op : list of lists containing a tuple and a float
        The list of singe and double excitation
        operators to consizer. represented in the form,
        [ [(p,q), t_pq], .... , [(p,q,s,r), t_pqrs], ... ]
        where p, q, r, s are idicies of normal ordered creation or anihilation
        operators.
    """

    # TODO(Nick): make sq_op a SQOperator object rather than a python list
    sq_op = []

    #for term, coeff in sorted(OF_fermiops.terms.items()):
    for term, coeff in OF_fermiops.terms.items():

        if (int(len(term) % 2) != 0):
            raise ValueError("OF_fermiops mush have equal number of annihilators and creators.")
        nbody = int(len(term)/2)
        sq_term = []
        sq_term_op = []

        if np.isclose(coeff, 0.0):
            continue

        for k, fermiop in enumerate(term): # a single annihilator or creator
            index, action = fermiop
            if k < nbody and action == 0:
                raise ValueError("OF_fermiops must have only normal ordered terms!")
            if k >= nbody and action == 1:
                raise ValueError("OF_fermiops must have only normal ordered terms!")

            sq_term_op.append(index)

        sq_term.append(tuple(sq_term_op))
        sq_term.append(coeff)
        sq_op.append(sq_term)

    return sq_op
