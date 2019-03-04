import qforte

def smart_print(Inputobj, print_type='compact'):

    """
    smart_print is a function that formatted and print instances of 
    several classes, including QuantumOperator, QuantumCircuit, and
    QuantumnComputer

    :param Inputobj: the input instance one want to print.
    :param print_type: print format, full or compact
    """
    
    if isinstance(Inputobj, qforte.QuantumOperator):
        print('\n Quantum operator:')

        if print_type == 'full':
            ops_term = Inputobj.terms()
            for term in ops_term:
                print(term[0])
                print('\n'.join(term[1].str()))
                print('\n')

        if print_type == 'compact':
            ops_term = Inputobj.terms()
            for term in ops_term:
                print(term[0], end="")
                strp = term[1].str()
                print('[', end="")
                for termstr in strp:
                    tmpstr = termstr.split('\n')
                    print(tmpstr[0][0], end="")
                    print(tmpstr[0][21], end=" ")
                print(']')

    if isinstance(Inputobj, qforte.QuantumCircuit):
        print('\n Quantum circuit:')

        if print_type == 'full':
            print('\n'.join(Inputobj.str()))

        if print_type == 'compact':
                strp = Inputobj.str()
                print('[', end="")
                for termstr in strp:
                    tmpstr = termstr.split('\n')
                    print(tmpstr[0][0], end="")
                    print(tmpstr[0][21], end=" ")
                print(']')

    if isinstance(Inputobj, qforte.QuantumComputer):
        print('\n Quantum Computer:')
        print('\n'.join(Inputobj.str()))

def build_circuit(Inputstr):
    circ = qforte.QuantumCircuit()
    sepstr = Inputstr.split()

    for i in lens(sepstr):
        numstr = ''
        for j in lens(sepstr)-1:
            numstr += sepstr[j+1]
            if sepstr[j+1] == ' ':
                break
        
        circ.add_gate(qforte.make_gate(sepstr[0], numstr))
    
    return circ
           
#def build_operator(Inputstr):

