import pytest
import qforte as qf
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
reference_tex = os.path.join(THIS_DIR, "circuit_to_latex_reference.tex")


class TestCircuitToLatex:
    def test_circ2latex_regression(self):
        circ = qf.Circuit()

        # For the purposes of this test, the values of the parameters defining
        # parametrized gates is irrelevant. The default of zero suffices.
        single_qubit_gates = ["X", "Y", "Z", "Rx", "Ry", "Rz", "H", "S", "T"]
        single_qubit_targets = [6, 1, 3, 5, 2, 4, 0, 2, 3]

        two_qubit_gates = ["cX", "CNOT", "acX", "aCNOT", "cZ"]
        two_qubit_targets = [5, 4, 2, 2, 1]
        two_qubit_controls = [1, 0, 0, 1, 6]

        for idx, gate in enumerate(single_qubit_gates):
            circ.add(qf.gate(gate, single_qubit_targets[idx]))

        for idx, gate in enumerate(two_qubit_gates):
            circ.add(qf.gate(gate, two_qubit_targets[idx], two_qubit_controls[idx]))

        circ.circuit_to_latex(filename="circuit_to_latex_generated")

        assert os.path.exists("circuit_to_latex_generated.tex")
        assert os.path.getsize("circuit_to_latex_generated.tex") > 0

        with open("circuit_to_latex_generated.tex", "r") as generated_file:
            generated_content = generated_file.read()
        with open(reference_tex, "r") as reference_file:
            reference_content = reference_file.read()
        assert generated_content == reference_content

        os.remove("circuit_to_latex_generated.tex")
