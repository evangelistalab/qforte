import qforte

def smart_print(Inputobj, print_type='compact'):

    """
    formatts and prints instances of
    several classes, including QubitOperator, Circuit, and
    QuantumnComputer

    :param Inputobj: (QubitOperator, Circuit, or
    QuantumnComputer) the input instance one want to print.

    :param print_type: (QubitOperator, Circuit, or
    QuantumnComputer) print format, full or compact
    """

    if isinstance(Inputobj, qforte.QubitOperator):
        print('\n Quantum operator:')

        if print_type == 'full':
            ops_term = Inputobj.terms()
            for term in ops_term:
                print(term[0])
                print('\n'.join(term[1].str()))
                print('\n')

        if print_type == 'compact':
            ops_term = Inputobj.terms()
            first = True
            for term in ops_term:
                if first:
                    first = False
                else:
                    print("+", end=" ")
                print(term[0], end="")
                print("[{}]".format(" ".join(term[1].str())))

    if isinstance(Inputobj, qforte.Circuit):
        print('\n Quantum circuit:')

        if print_type == 'full':
            print('\n'.join(Inputobj.str()))

        if print_type == 'compact':
            strp = Inputobj.str()
            print('[', end="")
            subfirst = True
            for termstr in strp:
                tmpstr = termstr.split('\n')
                tmp_a = tmpstr[0].split()
                if subfirst:
                    subfirst = False
                else:
                    print(" ", end=" ")
                print(tmp_a[0], end="") #Print the action string (X, Y, Z, cX, R, ...)
                tmp_b = tmpstr[0].split(':')
                control = tmp_b[2]
                tmp_c = tmp_b[1].split(',')
                target = tmp_c[0]
                if target == control: #Print the target and control(if necessary)
                    print(target, end="")
                else:
                    print(target, end="-")
                    print(control, end="")
            print(']')

    if isinstance(Inputobj, qforte.Computer):
        print('\n Quantum Computer:')
        print('\n'.join(Inputobj.str()))

"""
builds a Circuit conveniently from string based input

:param Inputstr: (string) the circuit to build, format:
['Action string']_['Target']_['Control(if needed)']_['Parameter(if needed)']
"""

def build_circuit(Inputstr):

    circ = qforte.Circuit()
    sepstr = Inputstr.split() #Separate string to a list by space

    for i in range(len(sepstr)):
        inputgate = sepstr[i].split('_')
        if len(inputgate) == 2:
            circ.add(qforte.gate(inputgate[0], int(inputgate[1]), int(inputgate[1])))
        else:
            if 'R' in inputgate[0]:
                circ.add(qforte.gate(inputgate[0], int(inputgate[1]), int(inputgate[1]), float(inputgate[2])))
            else:
                circ.add(qforte.gate(inputgate[0], int(inputgate[1]), int(inputgate[2])))

    return circ

"""
builds a Circuit
conveniently from input

:param Inputstr: (string) the operator to build, format:
['coeff1, circ1; coeff2, circ2, ...']
"""

def build_operator(Inputstr):

    ops = qforte.QubitOperator()
    sepstr = Inputstr.split(';')
    for i in range(len(sepstr)):
        inputterm = sepstr[i].split(',')
        ops.add(complex(inputterm[0]), qforte.build_circuit(inputterm[1]))

    return ops
