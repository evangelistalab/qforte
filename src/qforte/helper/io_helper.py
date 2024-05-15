import qforte


def smart_print(Inputobj):
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
        print("\n Quantum operator:")
        ops_term = Inputobj.terms()
        first = True
        for term in ops_term:
            if first:
                first = False
            else:
                print("+", end=" ")
            print(term[0], end=" ")
            strp = term[1].str()
            strp = strp[1:-1].split(" ")
            strp = " ".join(map(str, strp))
            print("(" + strp + ")", "|Ψ>")

    if isinstance(Inputobj, qforte.Circuit):
        print("\n Quantum circuit:")
        strp = Inputobj.str()
        strp = strp[1:-1].split(" ")
        strp = " ".join(map(str, strp))
        print("(" + strp + ")", "|Ψ>")

    if isinstance(Inputobj, qforte.Computer):
        print("\n Quantum Computer:")
        print("\n".join(Inputobj.str()))


"""
builds a Circuit conveniently from string based input

:param Inputstr: (string) the circuit to build, format:
['Action string']_['Target']_['Control(if needed)']_['Parameter(if needed)']
"""


def build_circuit(Inputstr):
    circ = qforte.Circuit()
    sepstr = Inputstr.split()  # Separate string to a list by space

    for i in range(len(sepstr)):
        inputgate = sepstr[i].split("_")
        if len(inputgate) == 2:
            circ.add(qforte.gate(inputgate[0], int(inputgate[1]), int(inputgate[1])))
        else:
            if "R" in inputgate[0]:
                circ.add(
                    qforte.gate(
                        inputgate[0],
                        int(inputgate[1]),
                        int(inputgate[1]),
                        float(inputgate[2]),
                    )
                )
            else:
                circ.add(
                    qforte.gate(inputgate[0], int(inputgate[1]), int(inputgate[2]))
                )

    return circ


"""
builds a Circuit
conveniently from input

:param Inputstr: (string) the operator to build, format:
['coeff1, circ1; coeff2, circ2, ...']
"""


def build_operator(Inputstr):
    ops = qforte.QubitOperator()
    sepstr = Inputstr.split(";")
    for i in range(len(sepstr)):
        inputterm = sepstr[i].split(",")
        ops.add(complex(inputterm[0]), qforte.build_circuit(inputterm[1]))

    return ops
