from pytest import approx
import qforte as qf
import numpy as np
import random

one_qubit_gate_pool = [
    "X",
    "Y",
    "Z",
    "Rx",
    "Ry",
    "Rz",
    "H",
    "S",
    "T",
    "R",
    "V",
    "adj(V)",
]
two_qubit_gate_pool = [
    "CNOT",
    "aCNOT",
    "cY",
    "cZ",
    "cV",
    "SWAP",
    "cRz",
    "cR",
    "A",
    "adj(cV)",
]

parametrized_gates = {"Rx", "Ry", "Rz", "R", "cR", "cRz", "A"}

parametrized_gate_periods = {
    "Rx": 4 * np.pi,
    "Ry": 4 * np.pi,
    "Rz": 4 * np.pi,
    "R": 2 * np.pi,
    "cR": 2 * np.pi,
    "cRz": 4 * np.pi,
    "A": 2 * np.pi,
}

diagonal_1qubit_gates = {"T", "S", "Z", "Rz", "R"}

phase_1qubit_gates = ["T", "S", "Z", "R"]

symmetrical_2qubit_gates = {"cZ", "cR", "SWAP"}

involutory_gates = {
    "X",
    "Y",
    "Z",
    "H",
    "CNOT",
    "cX",
    "aCNOT",
    "acX",
    "cY",
    "cZ",
    "SWAP",
}

pairs_of_commuting_1qubit_gates = {
    ("X", "X"),
    ("Rx", "X"),
    ("V", "X"),
    ("Y", "Y"),
    ("Ry", "Y"),
    ("Z", "Z"),
    ("S", "Z"),
    ("T", "Z"),
    ("Rz", "Z"),
    ("R", "Z"),
    ("H", "H"),
    ("S", "S"),
    ("S", "T"),
    ("Rz", "S"),
    ("R", "S"),
    ("T", "T"),
    ("Rz", "T"),
    ("R", "T"),
    ("Rx", "Rx"),
    ("Rx", "V"),
    ("Ry", "Ry"),
    ("Rz", "Rz"),
    ("R", "Rz"),
    ("R", "R"),
    ("V", "V"),
    ("adj(V)", "X"),
    ("adj(V)", "Rx"),
    ("adj(V)", "V"),
    ("adj(V)", "adj(V)"),
}

pairs_of_simplifiable_1qubit_gates = {
    ("X", "X"),
    ("Y", "Y"),
    ("Z", "Z"),
    ("H", "H"),
    ("S", "S"),
    ("T", "T"),
    ("Rx", "Rx"),
    ("Ry", "Ry"),
    ("Rz", "Rz"),
    ("R", "R"),
    ("V", "V"),
    ("S", "T"),
    ("T", "Z"),
    ("R", "T"),
    ("S", "Z"),
    ("R", "S"),
    ("R", "Z"),
    ("adj(V)", "V"),
    ("adj(V)", "adj(V)"),
}

controlled_2qubit_to_1qubit_gate = {
    "CNOT": "X",
    "cX": "X",
    "aCNOT": "X",
    "acX": "X",
    "cY": "Y",
    "cZ": "Z",
    "cRz": "Rz",
    "cR": "R",
    "cV": "V",
    "adj(cV)": "adj(V)",
}

square_root_gates = {"T", "S", "V", "cV", "adj(V)", "adj(cV)"}

simplify_square_root_gates = {
    "T": "S",
    "S": "Z",
    "V": "X",
    "cV": "cX",
    "adj(V)": "X",
    "adj(cV)": "cX",
}


class TestEvaluateGateInteraction:
    def test_disjoint_gates(self):
        # two single-qubit gates
        for i in range(len(one_qubit_gate_pool)):
            gatetype1 = one_qubit_gate_pool[i]
            if gatetype1 in parametrized_gates:
                period = parametrized_gate_periods[gatetype1]
                parameter = random.uniform(-period, period)
                gate1 = qf.gate(gatetype1, 0, 0, parameter)
            elif gatetype1 == "adj(V)":
                gate1 = qf.gate("V", 0)
                gate1 = gate1.adjoint()
            else:
                gate1 = qf.gate(gatetype1, 0)
            for j in range(i, len(one_qubit_gate_pool)):
                gatetype2 = one_qubit_gate_pool[j]
                if gatetype2 in parametrized_gates:
                    period = parametrized_gate_periods[gatetype2]
                    parameter = random.uniform(-period, period)
                    gate2 = qf.gate(gatetype2, 1, 1, parameter)
                elif gatetype2 == "adj(V)":
                    gate2 = qf.gate("V", 1)
                    gate2 = gate2.adjoint()
                else:
                    gate2 = qf.gate(gatetype2, 1)
                assert qf.evaluate_gate_interaction(gate1, gate2) == (True, 0)

        # single- and two-qubit gates
        for gatetype1 in one_qubit_gate_pool:
            if gatetype1 in parametrized_gates:
                period = parametrized_gate_periods[gatetype1]
                parameter = random.uniform(-period, period)
                gate1 = qf.gate(gatetype1, 0, 0, parameter)
            elif gatetype1 == "adj(V)":
                gate1 = qf.gate("V", 0)
                gate1 = gate1.adjoint()
            else:
                gate1 = qf.gate(gatetype1, 0)
            for gatetype2 in two_qubit_gate_pool:
                if gatetype2 in parametrized_gates:
                    period = parametrized_gate_periods[gatetype2]
                    parameter = random.uniform(-period, period)
                    gate2 = qf.gate(gatetype2, 1, 2, parameter)
                elif gatetype2 == "adj(cV)":
                    gate2 = qf.gate("cV", 1, 2)
                    gate2 = gate2.adjoint()
                else:
                    gate2 = qf.gate(gatetype2, 1, 2)
                assert qf.evaluate_gate_interaction(gate1, gate2) == (True, 0)

        # two two-qubit gates
        for i in range(len(two_qubit_gate_pool)):
            gatetype1 = two_qubit_gate_pool[i]
            if gatetype1 in parametrized_gates:
                period = parametrized_gate_periods[gatetype1]
                parameter = random.uniform(-period, period)
                gate1 = qf.gate(gatetype1, 0, 1, parameter)
            elif gatetype1 == "adj(cV)":
                gate1 = qf.gate("cV", 0, 1)
                gate1 = gate1.adjoint()
            else:
                gate1 = qf.gate(gatetype1, 0, 1)
            for j in range(i, len(two_qubit_gate_pool)):
                gatetype2 = two_qubit_gate_pool[j]
                if gatetype2 in parametrized_gates:
                    period = parametrized_gate_periods[gatetype2]
                    parameter = random.uniform(-period, period)
                    gate2 = qf.gate(gatetype2, 2, 3, parameter)
                elif gatetype2 == "adj(cV)":
                    gate2 = qf.gate("cV", 2, 3)
                    gate2 = gate2.adjoint()
                else:
                    gate2 = qf.gate(gatetype2, 2, 3)
                assert qf.evaluate_gate_interaction(gate1, gate2) == (True, 0)

    def test_commuting_1qubit_gates(self):
        for gatetype1, gatetype2 in pairs_of_commuting_1qubit_gates:
            if (gatetype1, gatetype2) in pairs_of_simplifiable_1qubit_gates:
                if (gatetype1, gatetype2) == ("adj(V)", "V"):
                    simplification_case = 4
                elif (
                    gatetype1 in phase_1qubit_gates
                    and gatetype2 in phase_1qubit_gates
                    and gatetype2 != gatetype1
                ):
                    simplification_case = 3
                else:
                    simplification_case = 1
            else:
                simplification_case = 0
            if gatetype1 in parametrized_gates:
                period = parametrized_gate_periods[gatetype1]
                parameter = random.uniform(-period, period)
                gate1 = qf.gate(gatetype1, 0, 0, parameter)
            elif gatetype1 == "adj(V)":
                gate1 = qf.gate("V", 0)
                gate1 = gate1.adjoint()
            else:
                gate1 = qf.gate(gatetype1, 0)
            if gatetype2 in parametrized_gates:
                period = parametrized_gate_periods[gatetype2]
                parameter = random.uniform(-period, period)
                gate2 = qf.gate(gatetype2, 0, 0, parameter)
            elif gatetype2 == "adj(V)":
                gate2 = qf.gate("V", 0)
                gate2 = gate2.adjoint()
            else:
                gate2 = qf.gate(gatetype2, 0)
            assert qf.evaluate_gate_interaction(gate1, gate2) == (
                True,
                simplification_case,
            )

    def test_non_commuting_1qubit_gates(self):
        for i in range(len(one_qubit_gate_pool)):
            gatetype1 = one_qubit_gate_pool[i]
            if gatetype1 in parametrized_gates:
                period = parametrized_gate_periods[gatetype1]
                parameter = random.uniform(-period, period)
                gate1 = qf.gate(gatetype1, 0, 0, parameter)
            elif gatetype1 == "adj(V)":
                gate1 = qf.gate("V", 0)
                gate1 = gate1.adjoint()
            else:
                gate1 = qf.gate(gatetype1, 0)
            for j in range(i, len(one_qubit_gate_pool)):
                gatetype2 = one_qubit_gate_pool[j]
                if (
                    tuple(sorted((gatetype1, gatetype2), key=str.lower))
                    in pairs_of_commuting_1qubit_gates
                ):
                    continue
                if gatetype2 in parametrized_gates:
                    period = parametrized_gate_periods[gatetype2]
                    parameter = random.uniform(-period, period)
                    gate2 = qf.gate(gatetype2, 0, 0, parameter)
                elif gatetype2 == "adj(V)":
                    gate2 = qf.gate("V", 0)
                    gate2 = gate2.adjoint()
                else:
                    gate2 = qf.gate(gatetype2, 0)
                assert qf.evaluate_gate_interaction(gate1, gate2) == (False, 0)

    def test_1qubit_and_2qubit_gate(self):
        for one_qubit_gate_type in one_qubit_gate_pool:
            if one_qubit_gate_type in parametrized_gates:
                period = parametrized_gate_periods[one_qubit_gate_type]
                parameter = random.uniform(-period, period)
                one_qubit_gate_target = qf.gate(one_qubit_gate_type, 0, 0, parameter)
                one_qubit_gate_control = qf.gate(one_qubit_gate_type, 1, 1, parameter)
            elif one_qubit_gate_type == "adj(V)":
                one_qubit_gate_target = qf.gate("V", 0)
                one_qubit_gate_target = one_qubit_gate_target.adjoint()
                one_qubit_gate_control = qf.gate("V", 1)
                one_qubit_gate_control = one_qubit_gate_control.adjoint()
            else:
                one_qubit_gate_target = qf.gate(one_qubit_gate_type, 0)
                one_qubit_gate_control = qf.gate(one_qubit_gate_type, 1)
            for two_qubit_gate_type in two_qubit_gate_pool:
                if two_qubit_gate_type in parametrized_gates:
                    period = parametrized_gate_periods[two_qubit_gate_type]
                    parameter = random.uniform(-period, period)
                    two_qubit_gate = qf.gate(two_qubit_gate_type, 0, 1, parameter)
                elif two_qubit_gate_type == "adj(cV)":
                    two_qubit_gate = qf.gate("cV", 0, 1)
                    two_qubit_gate = two_qubit_gate.adjoint()
                else:
                    two_qubit_gate = qf.gate(two_qubit_gate_type, 0, 1)
                print(one_qubit_gate_type, two_qubit_gate_type)
                if two_qubit_gate_type == "SWAP" or two_qubit_gate_type == "A":
                    assert qf.evaluate_gate_interaction(
                        one_qubit_gate_target, two_qubit_gate
                    ) == (False, 0)
                    assert qf.evaluate_gate_interaction(
                        one_qubit_gate_control, two_qubit_gate
                    ) == (False, 0)
                    continue
                if (
                    tuple(
                        sorted(
                            (
                                one_qubit_gate_type,
                                controlled_2qubit_to_1qubit_gate[two_qubit_gate_type],
                            ),
                            key=str.lower,
                        )
                    )
                    in pairs_of_commuting_1qubit_gates
                ):
                    assert qf.evaluate_gate_interaction(
                        one_qubit_gate_target, two_qubit_gate
                    ) == (True, 0)
                else:
                    assert qf.evaluate_gate_interaction(
                        one_qubit_gate_target, two_qubit_gate
                    ) == (False, 0)
                if one_qubit_gate_type in diagonal_1qubit_gates:
                    assert qf.evaluate_gate_interaction(
                        one_qubit_gate_control, two_qubit_gate
                    ) == (True, 0)
                else:
                    assert qf.evaluate_gate_interaction(
                        one_qubit_gate_control, two_qubit_gate
                    ) == (False, 0)

    def test_2qubit_gates(self):
        for i in range(len(two_qubit_gate_pool)):
            gatetype1 = two_qubit_gate_pool[i]
            if gatetype1 in parametrized_gates:
                period = parametrized_gate_periods[gatetype1]
                parameter = random.uniform(-period, period)
                gate1 = qf.gate(gatetype1, 0, 1, parameter)
            elif gatetype1 == "adj(cV)":
                gate1 = qf.gate("cV", 0, 1)
                gate1 = gate1.adjoint()
            else:
                gate1 = qf.gate(gatetype1, 0, 1)
            for j in range(i, len(two_qubit_gate_pool)):
                gatetype2 = two_qubit_gate_pool[j]
                if gatetype2 in parametrized_gates:
                    period = parametrized_gate_periods[gatetype2]
                    parameter = random.uniform(-period, period)
                    gate2a = qf.gate(gatetype2, 0, 2, parameter)
                    gate2b = qf.gate(gatetype2, 2, 1, parameter)
                    gate2c = qf.gate(gatetype2, 2, 0, parameter)
                    gate2d = qf.gate(gatetype2, 1, 2, parameter)
                    gate2e = qf.gate(gatetype2, 0, 1, parameter)
                    gate2f = qf.gate(gatetype2, 1, 0, parameter)
                elif gatetype2 == "adj(cV)":
                    gate2a = qf.gate("cV", 0, 2)
                    gate2b = qf.gate("cV", 2, 1)
                    gate2c = qf.gate("cV", 2, 0)
                    gate2d = qf.gate("cV", 1, 2)
                    gate2e = qf.gate("cV", 0, 1)
                    gate2f = qf.gate("cV", 1, 0)
                    gate2a = gate2a.adjoint()
                    gate2b = gate2b.adjoint()
                    gate2c = gate2c.adjoint()
                    gate2d = gate2d.adjoint()
                    gate2e = gate2e.adjoint()
                    gate2f = gate2f.adjoint()
                else:
                    gate2a = qf.gate(gatetype2, 0, 2)
                    gate2b = qf.gate(gatetype2, 2, 1)
                    gate2c = qf.gate(gatetype2, 2, 0)
                    gate2d = qf.gate(gatetype2, 1, 2)
                    gate2e = qf.gate(gatetype2, 0, 1)
                    gate2f = qf.gate(gatetype2, 1, 0)
                if "SWAP" in [gatetype1, gatetype2]:
                    assert qf.evaluate_gate_interaction(gate1, gate2a) == (False, 0)
                    assert qf.evaluate_gate_interaction(gate1, gate2b) == (False, 0)
                    assert qf.evaluate_gate_interaction(gate1, gate2c) == (False, 0)
                    assert qf.evaluate_gate_interaction(gate1, gate2d) == (False, 0)
                    if len({gatetype1, gatetype2}) == 1:
                        assert qf.evaluate_gate_interaction(gate1, gate2e) == (True, 2)
                        assert qf.evaluate_gate_interaction(gate1, gate2f) == (True, 2)
                    if len({gatetype1, gatetype2}) == 2:
                        if (
                            gatetype1 in symmetrical_2qubit_gates
                            and gatetype2 in symmetrical_2qubit_gates
                        ):
                            assert qf.evaluate_gate_interaction(gate1, gate2e) == (
                                True,
                                0,
                            )
                            assert qf.evaluate_gate_interaction(gate1, gate2f) == (
                                True,
                                0,
                            )
                        else:
                            assert qf.evaluate_gate_interaction(gate1, gate2e) == (
                                False,
                                0,
                            )
                            assert qf.evaluate_gate_interaction(gate1, gate2f) == (
                                False,
                                0,
                            )
                elif "A" in [gatetype1, gatetype2]:
                    assert qf.evaluate_gate_interaction(gate1, gate2a) == (False, 0)
                    assert qf.evaluate_gate_interaction(gate1, gate2b) == (False, 0)
                    assert qf.evaluate_gate_interaction(gate1, gate2c) == (False, 0)
                    assert qf.evaluate_gate_interaction(gate1, gate2d) == (False, 0)
                    if (
                        gatetype1 in symmetrical_2qubit_gates
                        or gatetype2 in symmetrical_2qubit_gates
                    ):
                        assert qf.evaluate_gate_interaction(gate1, gate2e) == (True, 0)
                        assert qf.evaluate_gate_interaction(gate1, gate2f) == (True, 0)
                    else:
                        assert qf.evaluate_gate_interaction(gate1, gate2e) == (False, 0)
                        assert qf.evaluate_gate_interaction(gate1, gate2f) == (False, 0)
                else:
                    if (
                        tuple(
                            sorted(
                                [
                                    controlled_2qubit_to_1qubit_gate[gatetype1],
                                    controlled_2qubit_to_1qubit_gate[gatetype2],
                                ],
                                key=str.lower,
                            )
                        )
                        in pairs_of_commuting_1qubit_gates
                    ):
                        assert qf.evaluate_gate_interaction(gate1, gate2a) == (True, 0)
                    else:
                        assert qf.evaluate_gate_interaction(gate1, gate2a) == (False, 0)
                    assert qf.evaluate_gate_interaction(gate1, gate2b) == (True, 0)
                    if (
                        controlled_2qubit_to_1qubit_gate[gatetype1]
                        in diagonal_1qubit_gates
                    ):
                        assert qf.evaluate_gate_interaction(gate1, gate2c) == (True, 0)
                    else:
                        assert qf.evaluate_gate_interaction(gate1, gate2c) == (False, 0)
                    if (
                        controlled_2qubit_to_1qubit_gate[gatetype2]
                        in diagonal_1qubit_gates
                    ):
                        assert qf.evaluate_gate_interaction(gate1, gate2d) == (True, 0)
                    else:
                        assert qf.evaluate_gate_interaction(gate1, gate2d) == (False, 0)
                    if (
                        gatetype1 in symmetrical_2qubit_gates
                        and gatetype2 in symmetrical_2qubit_gates
                    ):
                        assert qf.evaluate_gate_interaction(gate1, gate2e) == (
                            True,
                            2 * (gatetype1 == gatetype2),
                        )
                        assert qf.evaluate_gate_interaction(gate1, gate2f) == (
                            True,
                            2 * (gatetype1 == gatetype2),
                        )
                        continue
                    if (
                        tuple(
                            sorted(
                                [
                                    controlled_2qubit_to_1qubit_gate[gatetype1],
                                    controlled_2qubit_to_1qubit_gate[gatetype2],
                                ],
                                key=str.lower,
                            )
                        )
                        in pairs_of_commuting_1qubit_gates
                    ):
                        if tuple(
                            sorted(
                                [
                                    controlled_2qubit_to_1qubit_gate[gatetype1],
                                    controlled_2qubit_to_1qubit_gate[gatetype2],
                                ],
                                key=str.lower,
                            )
                        ) == ("adj(V)", "V"):
                            assert qf.evaluate_gate_interaction(gate1, gate2e) == (
                                True,
                                4,
                            )
                        else:
                            assert qf.evaluate_gate_interaction(gate1, gate2e) == (
                                True,
                                (gatetype1 == gatetype2),
                            )
                    else:
                        assert qf.evaluate_gate_interaction(gate1, gate2e) == (False, 0)
                    if (
                        controlled_2qubit_to_1qubit_gate[gatetype1]
                        in diagonal_1qubit_gates
                        and controlled_2qubit_to_1qubit_gate[gatetype2]
                        in diagonal_1qubit_gates
                    ):
                        assert qf.evaluate_gate_interaction(gate1, gate2f) == (True, 0)
                    else:
                        assert qf.evaluate_gate_interaction(gate1, gate2f) == (False, 0)


class TestCircuitSimplify:
    def test_simplify_involutory_gates(self):
        for gatetype in involutory_gates:
            if gatetype in one_qubit_gate_pool:
                circ = qf.Circuit()
                circ.add(qf.gate(gatetype, 0))
                circ.add(qf.gate(gatetype, 0))
                circ.simplify()
                assert circ.gates() == []
            elif gatetype in symmetrical_2qubit_gates:
                circ = qf.Circuit()
                circ.add(qf.gate(gatetype, 0, 1))
                circ.add(qf.gate(gatetype, 0, 1))
                circ.simplify()
                assert circ.gates() == []
                circ = qf.Circuit()
                circ.add(qf.gate(gatetype, 0, 1))
                circ.add(qf.gate(gatetype, 1, 0))
                circ.simplify()
                assert circ.gates() == []
            else:
                circ = qf.Circuit()
                circ.add(qf.gate(gatetype, 0, 1))
                circ.add(qf.gate(gatetype, 0, 1))
                circ.simplify()
                assert circ.gates() == []

    def test_simplify_parametrized_gates(self):
        for gatetype in parametrized_gates:
            if gatetype == "A":
                circ = qf.Circuit()
                circ.add(qf.gate(gatetype, 0, 1, 0.5))
                circ.add(qf.gate(gatetype, 0, 1, -0.2))
                circ.simplify()
                assert len(circ.gates()) == 2
                continue
            if gatetype in one_qubit_gate_pool:
                circ = qf.Circuit()
                circ.add(qf.gate(gatetype, 0, 0, 0.5))
                circ.add(qf.gate(gatetype, 0, 0, -0.2))
                circ.simplify()
                assert circ.size() == 1
                assert circ.gates()[0].gate_id() == gatetype
                assert circ.gates()[0].parameter() == 0.3
            elif gatetype in symmetrical_2qubit_gates:
                circ = qf.Circuit()
                circ.add(qf.gate(gatetype, 0, 1, 0.5))
                circ.add(qf.gate(gatetype, 0, 1, -0.2))
                circ.simplify()
                assert circ.size() == 1
                assert circ.gates()[0].gate_id() == gatetype
                assert circ.gates()[0].parameter() == 0.3
                circ = qf.Circuit()
                circ.add(qf.gate(gatetype, 0, 1, 0.5))
                circ.add(qf.gate(gatetype, 1, 0, -0.2))
                circ.simplify()
                assert circ.size() == 1
                assert circ.gates()[0].gate_id() == gatetype
                assert circ.gates()[0].parameter() == 0.3
            else:
                circ = qf.Circuit()
                circ.add(qf.gate(gatetype, 0, 1, 0.5))
                circ.add(qf.gate(gatetype, 0, 1, -0.2))
                circ.simplify()
                assert circ.size() == 1
                assert circ.gates()[0].gate_id() == gatetype
                assert circ.gates()[0].parameter() == 0.3

    def test_simplify_square_root_gates(self):
        for gatetype in square_root_gates:
            if gatetype in one_qubit_gate_pool:
                if gatetype == "adj(V)":
                    gate = qf.gate("V", 0)
                    gate = gate.adjoint()
                else:
                    gate = qf.gate(gatetype, 0)
            else:
                if gatetype == "adj(cV)":
                    gate = qf.gate("cV", 0, 1)
                    gate = gate.adjoint()
                else:
                    gate = qf.gate(gatetype, 0, 1)
            circ = qf.Circuit()
            circ.add(gate)
            circ.add(gate)
            circ.simplify()
            assert circ.size() == 1
            assert circ.gates()[0].gate_id() == simplify_square_root_gates[gatetype]

    def test_simplify_1qubit_phase_gates(slef):
        for i in range(len(phase_1qubit_gates)):
            gatetype1 = phase_1qubit_gates[i]
            if gatetype1 in parametrized_gates:
                period = parametrized_gate_periods[gatetype1]
                parameter1 = random.uniform(-period, period)
                gate1 = qf.gate(gatetype1, 0, parameter1)
            else:
                gate1 = qf.gate(gatetype1, 0)
            for j in range(i + 1, len(phase_1qubit_gates)):
                gatetype2 = phase_1qubit_gates[j]
                if gatetype2 in parametrized_gates:
                    period = parametrized_gate_periods[gatetype2]
                    parameter2 = random.uniform(-period, period)
                    gate2 = qf.gate(gatetype2, 0, parameter2)
                else:
                    gate2 = qf.gate(gatetype2, 0)
                circ1 = qf.Circuit()
                circ1.add(gate1)
                circ1.add(gate2)
                circ1.simplify()
                assert circ1.size() == 1
                assert circ1.gates()[0].gate_id() == "R"
                circ2 = qf.Circuit()
                circ2.add(gate2)
                circ2.add(gate1)
                circ2.simplify()
                assert circ2.size() == 1
                assert circ2.gates()[0].gate_id() == "R"

    def test_simplify_unitary_pairs(self):
        for gatetype in ["V", "cV", "adj(V)", "adj(cV)"]:
            if gatetype in one_qubit_gate_pool:
                if gatetype == "adj(V)":
                    gate = qf.gate("V", 0)
                    gate = gate.adjoint()
                elif gatetype in parametrized_gates:
                    period = parametrized_gate_periods[gatetype]
                    parameter = random.uniform(-period / 2, period / 2)
                    gate = qf.gate(gatetype, 0, 0, parameter)
                else:
                    gate = qf.gate(gatetype, 0)
            else:
                if gatetype == "adj(cV)":
                    gate = qf.gate("cV", 0, 1)
                    gate = gate.adjoint()
                elif gatetype in parametrized_gates:
                    period = parametrized_gate_periods[gatetype]
                    parameter = random.uniform(-period / 2, period / 2)
                    gate = qf.gate(gatetype, 0, 1, parameter)
                else:
                    gate = qf.gate(gatetype, 0, 1)
            gate_adjoint = gate.adjoint()
            circ = qf.Circuit()
            circ.add(gate)
            circ.add(gate_adjoint)
            circ.simplify()
            assert circ.size() == 0

    def test_simplify_H6_STO6G_UCCSDT_ansatz_circuit(self):
        Rhh = 2

        mol = qf.system_factory(
            system_type="molecule",
            build_type="psi4",
            basis="sto-6g",
            mol_geometry=[
                ("H", (0, 0, -5 * Rhh / 2)),
                ("H", (0, 0, -3 * Rhh / 2)),
                ("H", (0, 0, -Rhh / 2)),
                ("H", (0, 0, Rhh / 2)),
                ("H", (0, 0, 3 * Rhh / 2)),
                ("H", (0, 0, 5 * Rhh / 2)),
            ],
            symmetry="d2h",
            multiplicity=1,  # Only singlets will work with QForte
            charge=0,
            num_frozen_docc=0,
            num_frozen_uocc=0,
            run_mp2=0,
            run_ccsd=0,
            run_cisd=0,
            run_fci=0,
        )

        feb_uccsdt = qf.UCCNPQE(mol, compact_excitations=True, qubit_excitations=False)
        feb_uccsdt.run(
            pool_type="SDT", opt_maxiter=10, optimizer="jacobi", opt_thresh=1.0e-5
        )

        feb_ansatz_circ = feb_uccsdt.build_Uvqc()

        simplified_feb_ansatz_circ = qf.Circuit(feb_ansatz_circ)
        simplified_feb_ansatz_circ.simplify()

        qc = qf.Computer(mol.hamiltonian.num_qubits())
        qc.apply_circuit(feb_ansatz_circ)

        qc_simplified = qf.Computer(mol.hamiltonian.num_qubits())
        qc_simplified.apply_circuit(simplified_feb_ansatz_circ)

        assert feb_ansatz_circ.size() == 7896
        assert feb_ansatz_circ.get_num_cnots() == 4654
        assert simplified_feb_ansatz_circ.size() == 7364
        assert simplified_feb_ansatz_circ.get_num_cnots() == 4132
        assert (
            qc_simplified.direct_op_exp_val(mol.hamiltonian)
            - qc.direct_op_exp_val(mol.hamiltonian)
            == 0
        )
        assert (
            np.max(
                np.abs(
                    np.array(qc_simplified.get_coeff_vec())
                    - np.array(qc.get_coeff_vec())
                )
            )
            < 1.0e-14
        )
