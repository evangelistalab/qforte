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
                trigger = 0 # A flag for anormallies
                print('[', end="")
                for termstr in strp:
                    tmpstr = termstr.split('\n')
                    tmp_a = tmpstr[0].split()
                    print(tmp_a[0], end="") #Print the action string (X, Y, Z, cX, R, ...)
                    if str(tmp_a[0]) == 'R':
                        trigger = 1
                    tmp_b = tmpstr[0].split(':')
                    control = tmp_b[2]
                    tmp_c = tmp_b[1].split(',')
                    target = tmp_c[0]
                    if target == control: #Print the target and control(if necessary)
                        print(target, end=" ")
                    else:
                        print(target, end="-")
                        print(control, end=" ")
                print(']')
                if trigger == 1:
                    print('R gate presented, use \'full\' print to see the matrix! \n')

    if isinstance(Inputobj, qforte.QuantumCircuit):
        print('\n Quantum circuit:')

        if print_type == 'full':
            print('\n'.join(Inputobj.str()))

        if print_type == 'compact':
                strp = Inputobj.str()
                print('[', end="")
                for termstr in strp:
                    tmpstr = termstr.split('\n')
                    tmp_a = tmpstr[0].split()
                    print(tmp_a[0], end="") #Print the action string (X, Y, Z, cX, R, ...)
                    tmp_b = tmpstr[0].split(':')
                    control = tmp_b[2]
                    tmp_c = tmp_b[1].split(',')
                    target = tmp_c[0]
                    if target == control: #Print the target and control(if necessary)
                        print(target, end=" ")
                    else:
                        print(target, end="-")
                        print(control, end=" ")
                print(']')

    if isinstance(Inputobj, qforte.QuantumComputer):
        print('\n Quantum Computer:')
        print('\n'.join(Inputobj.str()))

def build_circuit(Inputstr):

    """
    build_circuit is a function that build a QuantumCircuit 
    conveniently from input

    :param Inputstr: the circuit to build, format:
    ['Action string']_['Target']_['Control(if needed)']_['Parameter']
    """

    circ = qforte.QuantumCircuit()
    sepstr = Inputstr.split() #Separate string to a list by space

    for i in range(len(sepstr)):
        inputgate = sepstr[i].split('_')
        if len(inputgate) == 2:
            circ.add_gate(qforte.make_gate(inputgate[0], int(inputgate[1]), int(inputgate[1])))
        else:
            if inputgate[0] == 'R':
                circ.add_gate(qforte.make_gate(inputgate[0], int(inputgate[1]), int(inputgate[1]), float(inputgate[2])))
            else:
                circ.add_gate(qforte.make_gate(inputgate[0], int(inputgate[1]), int(inputgate[2])))
        
    return circ
           

