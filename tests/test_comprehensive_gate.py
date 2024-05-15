from pytest import approx
from qforte import Circuit, Computer, gate
import numpy as np

num_qubits = 5
prep_circ = Circuit()
ct_lst = [
    (4, 3),
    (4, 2),
    (4, 1),
    (4, 0),
    (3, 2),
    (3, 1),
    (3, 0),
    (2, 1),
    (2, 0),
    (1, 0),
]

for i in range(num_qubits):
    prep_circ.add(gate("H", i, i))

for i in range(num_qubits):
    prep_circ.add(gate("cR", i, i + 1, 1.116 / (i + 1.0)))


def generic_test_circ_vec_builder(qb_list, id):
    circ_vec_tc = [Circuit() for i in range(len(qb_list))]
    circ_vec_ct = [Circuit() for i in range(len(qb_list))]
    for i, pair in enumerate(ct_lst):
        t = pair[0]
        c = pair[1]
        if id in ["A", "cR", "cRz"]:
            circ_vec_ct[i].add(gate(id, t, c, 3.17 * t * c))
            circ_vec_tc[i].add(gate(id, c, t, 1.41 * t * c))

        else:
            circ_vec_ct[i].add(gate(id, t, c))
            circ_vec_tc[i].add(gate(id, c, t))

    return circ_vec_tc, circ_vec_ct


def circuit_tester(prep, test_circ):
    for gate in test_circ.gates():
        id = gate.gate_id()
        target = gate.target()
        control = gate.control()

        num_qubits = 5

        qc1 = Computer(num_qubits)
        qc2 = Computer(num_qubits)

        qc1.apply_circuit_safe(prep)
        qc2.apply_circuit_safe(prep)

        qc1.apply_gate_safe(gate)
        qc2.apply_gate(gate)

        C1 = np.array(qc1.get_coeff_vec())
        C2 = np.array(qc2.get_coeff_vec())

        diff_norm = np.linalg.norm(C1 - C2)

        if diff_norm != (0.0 + 0.0j):
            print("|C - C_safe|F^2   control   target   id")
            print("----------------------------------------")
            print(diff_norm, "              ", control, "       ", target, "      ", id)

        return diff_norm


class TestComprehensiveGates:
    def test_gates(self):
        gate_ids = ["cV", "cX", "acX", "cY", "cZ", "cR", "cRz", "SWAP", "A"]

        for id in gate_ids:
            circ_tc, circ_ct = generic_test_circ_vec_builder(ct_lst, id)

            print("\n-------------------")
            print("Testing " + id + " circuits")
            print("-------------------")

            for circ in circ_ct:
                ct_val = circuit_tester(prep_circ, circ)
                assert ct_val == approx(0, abs=1.0e-16)

            for circ in circ_tc:
                tc_val = circuit_tester(prep_circ, circ)
                assert tc_val == approx(0, abs=1.0e-16)
