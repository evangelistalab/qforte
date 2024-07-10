import qforte


def circuit_to_latex(circ, filename="circuit", max_circuit_depth_per_tikz=20):
    """
    This function constructs the filename.tex latex
    file with the graphical representation of the given
    quantum circuit.

    Depending on how large the circuit is,
    compiling with lualatex is advisable.

    The quantikz package is required for
    compiling the tex file.

    Arguments
    =========

    circ: Circuit object
        The quantum circuit that we want to plot.

    filename: string
        The name of the latex file. The ".tex"
        file extension is not necessary.

    max_circuit_depth_per_tikz: int
        The maximum circuit dpeth for a given tikz
        drawing. To improve compilation efficiency,
        instead of generating a single, potentially
        large, tikz drawing, the circuit is split into
        multiple tikz figures, each containing at most
        max_circuit_depth_per_tikz layers.
        WARNING: Setting this parameter to a large
        number can significantly increase the latex
        compilation time.

    Returns
    =======

    filename.tex: latex file
        The latex file containing the plot of the
        given quantum circuit.
    """

    if not isinstance(filename, str):
        raise TypeError("The filename needs to be a string!")

    if (
        not isinstance(max_circuit_depth_per_tikz, int)
        or max_circuit_depth_per_tikz <= 0
    ):
        raise ValueError(
            "The maximum circuit depth per tikz figure needs to be a positive integer!"
        )

    print(
        "WARNING 1: The quantikz package is required for compiling the generated tex file!"
    )
    print("WARNING 2: For large quantum circuits, compiling with lualatex is advised!")

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

    if filename.endswith(".tex"):
        texfile = open(filename, "w")
    else:
        texfile = open(filename + ".tex", "w")
    texfile.write(preamble)
    texfile.write(tikz_start)

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
        wires[wire] = "            \lstick{$\ket{q_{" + str(wire) + "}}$}"
        gate_count[wire] = 0

    for gate in circ.gates():
        # To help the tex compilation, large quantum circuits are split
        # into smaller ones. If the maximum number of allowed gates in a
        # given wire has been reached and a new gate will be added to that
        # wire, create a new tikz picture
        current_max_circuit_depth = max(gate_count.values())
        if current_max_circuit_depth == max_circuit_depth_per_tikz:
            wires_with_max_circuit_depth_per_tikz = [
                wire
                for wire in gate_count
                if gate_count[wire] == max_circuit_depth_per_tikz
            ]
            if (
                gate.target() in wires_with_max_circuit_depth_per_tikz
                or gate.control() in wires_with_max_circuit_depth_per_tikz
            ):
                for wire in wires:
                    for i in range(max_circuit_depth_per_tikz - gate_count[wire]):
                        wires[wire] += " & \qw"
                for wire in reversed(list(qubit_ids)[1:]):
                    texfile.write(wires[wire] + " \\\ \n")
                wire = list(qubit_ids)[0]
                texfile.write(wires[wire] + "\n")
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
            # To avoid overlapping gates, the gate count of
            # the target and control qubits as well as all
            # qubits inbetween are shifted accordingly.
            if gate.target() > gate.control():
                max_qubit_id = gate.target()
                min_qubit_id = gate.control()
            else:
                max_qubit_id = gate.control()
                min_qubit_id = gate.target()
            max_depth = 0
            for wire in range(min_qubit_id, max_qubit_id + 1):
                if max_depth < gate_count[wire]:
                    max_depth = gate_count[wire]
            for wire in range(min_qubit_id, max_qubit_id + 1):
                for i in range(max_depth - gate_count[wire]):
                    wires[wire] += " & \qw"
                    gate_count[wire] += 1
            diff = int(gate.control() - gate.target())
            if gate.gate_id() not in ["cZ", "aCNOT", "acX", "CNOT", "cX"]:
                raise ValueError(
                    "The only two-qubit gates that are currently supported are: CNOT, aCNOT, and cZ!"
                )
            if gate.gate_id() == "cZ":
                wires[gate.target()] += " & \\ctrl{}"
            else:
                wires[gate.target()] += " & \\targ{}"
            if gate.gate_id() in {"aCNOT", "acX"}:
                wires[gate.control()] += " & \\octrl{" + str(diff) + "}"
            else:
                wires[gate.control()] += " & \\ctrl{" + str(diff) + "}"
            gate_count[gate.target()] += 1
            gate_count[gate.control()] += 1
            # To avoid overlapping gates, the "void" between the
            # control and target qubits is filled with "\qw"
            for wire in range(min_qubit_id + 1, max_qubit_id):
                wires[wire] += " & \qw"
                gate_count[wire] += 1

    # All wires in the latex file need to have the same length. This is accomplished
    # by appending \qw to the shorter wires
    max_gate_count = max(gate_count.values())
    if max_gate_count > 0:
        for wire in wires:
            for i in range(max_gate_count - gate_count[wire]):
                wires[wire] += " & \qw"

        for wire in reversed(list(qubit_ids)[1:]):
            texfile.write(wires[wire] + " \\\ \n")
        wire = list(qubit_ids)[0]
        texfile.write(wires[wire] + "\n")
        texfile.write(tikz_end)
    texfile.write(epilogue)
    texfile.close()


qforte.Circuit.circuit_to_latex = circuit_to_latex
