from pytest import approx
from qforte import Computer, build_circuit, Toffoli, build_operator, Fredkin


class TestAdvGate:
    def test_advanced_gates(self):
        print("\n")
        trial_state = Computer(4)
        trial_circ = build_circuit("X_0 X_1")
        trial_state.apply_circuit(trial_circ)

        # verify Toffoli gate
        T_circ = Toffoli(0, 1, 2)
        print(T_circ.str())
        trial_state.apply_circuit(T_circ)  # This should turn the state to 1110
        a1_dag_a2 = build_operator("1.0, Z_2")
        exp = trial_state.direct_op_exp_val(a1_dag_a2)
        assert exp == approx(-1, abs=9e-16)  # Measure qubit 2 should give -1

        # verify Fredkin gate
        F_circ = Fredkin(1, 2, 3)
        print(F_circ.str())
        trial_state.apply_circuit(F_circ)  # This should turn the state to 1101
        # trial_state.apply_circuit_safe(F_circ) # This should turn the state to 1101
        a1_dag_a2 = build_operator("1.0, Z_2")
        exp = trial_state.direct_op_exp_val(a1_dag_a2)
        assert exp == approx(1, abs=9e-16)  # Measure qubit 2 should give +1
