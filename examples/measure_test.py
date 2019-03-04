import qforte #as qf
#from qforte import QuantumComputer as qforte.QuantumComputer
#from qforte import QuantumCircuit as qcirc
#from qforte import QuantumOperator as qo

bell_state = qforte.QuantumComputer(2)
bell_prep = [None]*2
bell_prep[0] = qforte.make_gate('H',0,0)
bell_prep[1] = qforte.make_gate('cX',1,0)
bell_circ = qforte.QuantumCircuit()

for gate in bell_prep:
    bell_circ.add_gate(gate)

print('bell state')
bell_state.apply_circuit(bell_circ)
bell_state.str()

I0 = qforte.make_gate('X',0,0)
Z0 = qforte.make_gate('X',1,1)

circ1 = qforte.QuantumCircuit()
circ1.add_gate(I0)

circ2 = qforte.QuantumCircuit()
circ2.add_gate(Z0)

n0 = qforte.QuantumOperator()
n0.add_term(0.5,circ1)
n0.add_term(0.5,circ2)

exp = bell_state.direct_op_exp_val(n0)
print(exp)

val_1 = bell_state.measure_circuit(circ1, 1000)
val_2 = bell_state.measure_circuit(circ2, 1000)
#print(val_1)
#print(val_2)

bell_state.str()

a = sum(val_1)/1000
b = sum(val_2)/1000

print(a)
print(b)


