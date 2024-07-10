from pytest import approx
from qforte import Computer, build_circuit, build_operator, qft, rev_qft


class TestQFT:
    def test_qft(self):
        trial_state = Computer(4)
        trial_circ = build_circuit("X_0 X_1")
        trial_state.apply_circuit(trial_circ)

        # verify direct transformation
        qft(trial_state, 0, 3)

        a1_dag_a2 = build_operator("1.0, Z_0")
        exp = trial_state.direct_op_exp_val(a1_dag_a2)
        assert exp == approx(0, abs=1.0e-16)

        # test unitarity
        qft(trial_state, 0, 2)
        rev_qft(trial_state, 0, 2)

        a1_dag_a2 = build_operator("1.0, Z_0")
        exp = trial_state.direct_op_exp_val(a1_dag_a2)
        assert exp == approx(0, abs=1.0e-16)

        # test reverse transformation
        qft(trial_state, 0, 3)

        a1_dag_a2 = build_operator("1.0, Z_0")
        exp = trial_state.direct_op_exp_val(a1_dag_a2)
        assert exp == approx(1.0, abs=1.0e-14)
