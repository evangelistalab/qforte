import qforte

def circuit_to_latex(circ):
    """
    This function constructs a latex file with
    the graphical representation of the given
    quantum circuit.

    Depending on how large the circuit is,
    compiling with lualatex is advisable.

    The quantikz package is required for
    compiling the tex file.

    Arguments
    =========

    circ: Circuit object
        The quantum circuit that we want to plot.

    Returns
    =======

    circ.tex: latex file
        The latex file containing the plot of the
        given quantum circuit.
    """

    preamble = r"""\documentclass[tikz]{standalone}

\usepackage{quantikz}

\begin{document}
"""

    tikz_start = r"""
\begin{tikzpicture}
    \node {
        \begin{quantikz}
"""

    tikz_end = r"""        \end{quantikz}
    };
\end{tikzpicture}
"""

    epilogue = r"""
\end{document}"""

    texfile = open("circuit.tex", "w")
    texfile.write(preamble)
    texfile.write(tikz_start)

    # The maximum gate count for a given tikz drawing.
    # This makes compilation easier
    max_gate_count_per_tikz = 20

    # set that contains the indices of the qubits appearing in circ
    qubit_ids = set()
    # dictionary of the form {qubit_id : wire}, where wire is the
    # latex code representing the quantum wire qubit_id
    wires = {}
    # dictionary of the form {qubit_id : gate_count}, where gate_count
    # is the number of gates appearing in quantum wire qubit_id
    gate_count = {}

    for gate in circ.gates():
        qubit_ids.add(gate.target())
        qubit_ids.add(gate.control())

    # initialize wires and gate_count dictionaries
    for wire in qubit_ids:
        wires[wire] = "            \lstick{$\ket{q_{"+str(wire)+"}}$}"
        gate_count[wire] = 0

    for gate in circ.gates():
        # To help the tex compilation, large quantum circuits are split
        # into smaller ones. If the maximum number of allowed gates in a
        # given wire has been reached and a new gate will be added to that
        # wire, create a new tikz picture
        current_max_gate_count = max(gate_count.values())
        if current_max_gate_count == max_gate_count_per_tikz:
            wires_with_max_gate_count_per_tikz = [wire for wire in gate_count if gate_count[wire] == max_gate_count_per_tikz]
            if gate.target() in wires_with_max_gate_count_per_tikz or gate.control() in wires_with_max_gate_count_per_tikz:
                for wire in wires:
                    for i in range(max_gate_count_per_tikz - gate_count[wire]):
                        wires[wire] += " & \qw"
                for wire in reversed(list(qubit_ids)[1:]):
                    texfile.write(wires[wire]+" \\\ \n")
                wire = list(qubit_ids)[0]
                texfile.write(wires[wire]+"\n")
                texfile.write(tikz_end)
                texfile.write(tikz_start)
                for wire in qubit_ids:
                    wires[wire] = "            \ \ldots \ \qw"
                    gate_count[wire] = 0
        # check if gate is acting on single qubit
        if gate.target() == gate.control():
            wires[gate.target()] += " & \gate{" + gate.gate_id() + "}"
            gate_count[gate.target()] += 1
        else:
            # currently QForte supports up to two-qubit gates
            count_target = gate_count[gate.target()]
            count_control = gate_count[gate.control()]
            # if the target and control wires contain a different
            # number of gates, we need to add "\qw" to equalize them
            if count_target < count_control:
                for i in range(count_control - count_target):
                    wires[gate.target()] += " & \qw"
                    gate_count[gate.target()] += 1
            if count_target > count_control:
                for i in range(count_target - count_control):
                    wires[gate.control()] += " & \qw"
                    gate_count[gate.control()] += 1
            diff = int(gate.control() - gate.target())
            if gate.gate_id() not in ["cZ", "aCNOT", "acX", "CNOT", "cX"]:
                raise ValueError("The only two-qubit gates that are currently supported are: CNOT, aCNOT, and cZ!")
            if gate.gate_id() == "cZ":
                wires[gate.target()] += " & \\ctrl{}"
            else:
                wires[gate.target()] += " & \\targ{}"
            if gate.gate_id() == 'aCNOT':
                wires[gate.control()] += " & \\octrl{"+str(diff)+"}"
            else:
                wires[gate.control()] += " & \\ctrl{"+str(diff)+"}"
            gate_count[gate.target()] += 1
            gate_count[gate.control()] += 1

    # All wires in the latex file need to have the same length. This is accomplished
    # by appending \qw to the shorter wires
    max_gate_count = max(gate_count.values())
    if max_gate_count > 0:
        for wire in wires:
            for i in range(max_gate_count - gate_count[wire]):
                wires[wire] += " & \qw"

        for wire in reversed(list(qubit_ids)[1:]):
            texfile.write(wires[wire]+" \\\ \n")
        wire = list(qubit_ids)[0]
        texfile.write(wires[wire]+"\n")
        texfile.write(tikz_end)
    texfile.write(epilogue)
    texfile.close()

qforte.Circuit.circuit_to_latex = circuit_to_latex
