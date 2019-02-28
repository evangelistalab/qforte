import qforte

def smart_print(Inputobj):

    """
    smart_print is a function that formatted and print instances of 
    several classes, including QuantumOperator, QuantumCircuit, and
    QuantumnComputer

    :param Inputobj: the input instance one want to print.
    """
    
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

    if isinstance(Inputobj, qforte.QuantumComputer):
        print('\n Quantum Computer:')
        print('\n'.join(Inputobj.str()))
