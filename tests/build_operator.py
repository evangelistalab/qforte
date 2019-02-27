import qforte
#from openfermion.ops import QubitOperator

trial_state = qforte.QuantumComputer(4)

trial_prep = [None]*5
trial_prep[0] = qforte.make_gate('H',0,0)
trial_prep[1] = qforte.make_gate('H',1,1)
trial_prep[2] = qforte.make_gate('H',2,2)
trial_prep[3] = qforte.make_gate('H',3,3)
trial_prep[4] = qforte.make_gate('cX',0,1)

trial_circ = qforte.QuantumCircuit()

#prepare the circuit
for gate in trial_prep:
    trial_circ.add_gate(gate)

# use circuit to prepare trial state
trial_state.apply_circuit(trial_circ)

test_operator = qforte.QubitOperator('X2 Y1', 0.0-0.25j)
test_operator += qforte.QubitOperator('Y2 Y1', 0.25)
test_operator += qforte.QubitOperator('X2 X1', 0.25)
test_operator += qforte.QubitOperator('Y2 X1', 0.0+0.25j)
print(test_operator)

qforte_operator = qforte.build_from_openferm(test_operator)

qforte.smart_print(qforte_operator)

exp = trial_state.direct_op_exp_val(qforte_operator)
print(exp)
