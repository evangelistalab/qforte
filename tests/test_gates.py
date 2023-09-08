import pytest
from pytest import approx
from qforte import Computer, Circuit, gate, QubitBasis, QubitOperator

# this function creates a QubitBasis object from a string representation
def make_basis(str):
    return QubitBasis(int(str[::-1], 2))

class TestGates:
    def test_X_gate(self):
        # test the Pauli X gate
        nqubits = 1
        basis0 = make_basis('0')
        basis1 = make_basis('1')
        computer = Computer(nqubits)
        X = gate('X',0);
        # test X|0> = |1>
        computer.apply_gate(X)
        coeff0 = computer.coeff(basis0)
        coeff1 = computer.coeff(basis1)
        assert coeff0 == approx(0, abs=1.0e-16)
        assert coeff1 == approx(1, abs=1.0e-16)
        # test X|1> = |0>
        computer.set_state([(basis1,1.0)])
        computer.apply_gate(X)
        coeff0 = computer.coeff(basis0)
        coeff1 = computer.coeff(basis1)
        assert coeff0 == approx(1, abs=1.0e-16)
        assert coeff1 == approx(0, abs=1.0e-16)


    def test_Y_gate(self):
        # test the Pauli Y gate
        nqubits = 1
        basis0 = make_basis('0')
        basis1 = make_basis('1')
        computer = Computer(nqubits)
        Y = gate('Y',0,0);
        # test Y|0> = i|1>
        computer.apply_gate(Y)
        coeff0 = computer.coeff(basis0)
        coeff1 = computer.coeff(basis1)
        assert coeff0 == approx(0, abs=1.0e-16)
        assert coeff1.imag == approx(1.0, abs=1.0-16)
        # test Y|1> = -i|0>
        computer.set_state([(basis1,1.0)])
        computer.apply_gate(Y)
        coeff0 = computer.coeff(basis0)
        coeff1 = computer.coeff(basis1)
        assert coeff0.imag == approx(-1, abs=1.0e-16)
        assert coeff1 == approx(0, abs=1.0e-16)


    def test_Z_gate(self):
        # test the Pauli Y gate
        nqubits = 1
        basis0 = make_basis('0')
        basis1 = make_basis('1')
        computer = Computer(nqubits)
        Z = gate('Z',0,0);
        # test Z|0> = |0>
        computer.apply_gate(Z)
        coeff0 = computer.coeff(basis0)
        coeff1 = computer.coeff(basis1)
        assert coeff0 == approx(1, abs=1.0e-16)
        assert coeff1 == approx(0, abs=1.0e-16)
        # test Z|1> = -|1>
        computer.set_state([(basis1,1.0)])
        computer.apply_gate(Z)
        coeff0 = computer.coeff(basis0)
        coeff1 = computer.coeff(basis1)
        assert coeff0 == approx(0, abs=1.0e-16)
        assert coeff1 == approx(-1.0, abs=1.0e-16)

    def test_cX_gate(self):
        # test the cX/CNOT gate
        nqubits = 2
        basis0 = make_basis('00') # basis0:|00>
        basis1 = make_basis('01') # basis1:|10>
        basis2 = make_basis('10') # basis2:|01>
        basis3 = make_basis('11') # basis3:|11>
        computer = Computer(nqubits)
        CNOT = gate('CNOT',0,1);

        # test CNOT|00> = |00>
        computer.set_state([(basis0,1.0)])
        computer.apply_gate(CNOT)
        coeff0 = computer.coeff(basis0)
        coeff1 = computer.coeff(basis1)
        coeff2 = computer.coeff(basis2)
        coeff3 = computer.coeff(basis3)
        assert coeff0 == approx(1.0, abs=1.0e-16)
        assert coeff1 == approx(0, abs=1.0e-16)
        assert coeff2 == approx(0, abs=1.0e-16)
        assert coeff3 == approx(0, abs=1.0e-16)

        # test CNOT|10> = |11>
        computer.set_state([(basis1,1.0)])
        computer.apply_gate(CNOT)
        coeff0 = computer.coeff(basis0)
        coeff1 = computer.coeff(basis1)
        coeff2 = computer.coeff(basis2)
        coeff3 = computer.coeff(basis3)
        assert coeff0 == approx(0, abs=1.0e-16)
        assert coeff1 == approx(0, abs=1.0e-16)
        assert coeff2 == approx(0, abs=1.0e-16)
        assert coeff3 == approx(1, abs=1.0e-16)

        # test CNOT|01> = |01>
        computer.set_state([(basis2,1.0)])
        computer.apply_gate(CNOT)
        coeff0 = computer.coeff(basis0)
        coeff1 = computer.coeff(basis1)
        coeff2 = computer.coeff(basis2)
        coeff3 = computer.coeff(basis3)
        assert coeff0 == approx(0, abs=1.0e-16)
        assert coeff1 == approx(0, abs=1.0e-16)
        assert coeff2 == approx(1, abs=1.0e-16)
        assert coeff3 == approx(0, abs=1.0e-16)

        # test CNOT|11> = |10>
        computer.set_state([(basis3,1.0)])
        computer.apply_gate(CNOT)
        coeff0 = computer.coeff(basis0)
        coeff1 = computer.coeff(basis1)
        coeff2 = computer.coeff(basis2)
        coeff3 = computer.coeff(basis3)
        assert coeff0 == approx(0, abs=1.0e-16)
        assert coeff1 == approx(1, abs=1.0e-16)
        assert coeff2 == approx(0, abs=1.0e-16)
        assert coeff3 == approx(0, abs=1.0e-16)

        with pytest.raises(ValueError):
            gate('CNOT',0,1.0)

    def test_acX_gate(self):
        # test the acX/aCNOT gate
        nqubits = 2
        basis0 = make_basis('00') # basis0:|00>
        basis1 = make_basis('01') # basis1:|10>
        basis2 = make_basis('10') # basis2:|01>
        basis3 = make_basis('11') # basis3:|11>
        computer = Computer(nqubits)
        aCNOT = gate('aCNOT',0,1);

        # test aCNOT|00> = |01>
        computer.set_state([(basis0,1.0)])
        computer.apply_gate(aCNOT)
        coeff0 = computer.coeff(basis0)
        coeff1 = computer.coeff(basis1)
        coeff2 = computer.coeff(basis2)
        coeff3 = computer.coeff(basis3)
        assert coeff0 == approx(0, abs=1.0e-16)
        assert coeff1 == approx(0, abs=1.0e-16)
        assert coeff2 == approx(1, abs=1.0e-16)
        assert coeff3 == approx(0, abs=1.0e-16)

        # test aCNOT|10> = |10>
        computer.set_state([(basis1,1.0)])
        computer.apply_gate(aCNOT)
        coeff0 = computer.coeff(basis0)
        coeff1 = computer.coeff(basis1)
        coeff2 = computer.coeff(basis2)
        coeff3 = computer.coeff(basis3)
        assert coeff0 == approx(0, abs=1.0e-16)
        assert coeff1 == approx(1, abs=1.0e-16)
        assert coeff2 == approx(0, abs=1.0e-16)
        assert coeff3 == approx(0, abs=1.0e-16)

        # test aCNOT|01> = |00>
        computer.set_state([(basis2,1.0)])
        computer.apply_gate(aCNOT)
        coeff0 = computer.coeff(basis0)
        coeff1 = computer.coeff(basis1)
        coeff2 = computer.coeff(basis2)
        coeff3 = computer.coeff(basis3)
        assert coeff0 == approx(1, abs=1.0e-16)
        assert coeff1 == approx(0, abs=1.0e-16)
        assert coeff2 == approx(0, abs=1.0e-16)
        assert coeff3 == approx(0, abs=1.0e-16)

        # test aCNOT|11> = |11>
        computer.set_state([(basis3,1.0)])
        computer.apply_gate(aCNOT)
        coeff0 = computer.coeff(basis0)
        coeff1 = computer.coeff(basis1)
        coeff2 = computer.coeff(basis2)
        coeff3 = computer.coeff(basis3)
        assert coeff0 == approx(0, abs=1.0e-16)
        assert coeff1 == approx(0, abs=1.0e-16)
        assert coeff2 == approx(0, abs=1.0e-16)
        assert coeff3 == approx(1, abs=1.0e-16)

        with pytest.raises(ValueError):
            gate('aCNOT',0,1.0)

    def test_cY_gate(self):
        # test the cY gate
        nqubits = 2
        basis0 = make_basis('00') # basis0:|00>
        basis1 = make_basis('01') # basis1:|10>
        basis2 = make_basis('10') # basis2:|01>
        basis3 = make_basis('11') # basis3:|11>
        computer = Computer(nqubits)
        cY = gate('cY',0,1);

        # test cY|00> = |00>
        computer.set_state([(basis0,1.0)])
        computer.apply_gate(cY)
        coeff0 = computer.coeff(basis0)
        coeff1 = computer.coeff(basis1)
        coeff2 = computer.coeff(basis2)
        coeff3 = computer.coeff(basis3)
        assert coeff0 == approx(1, abs=1.0e-16)
        assert coeff1 == approx(0, abs=1.0e-16)
        assert coeff2 == approx(0, abs=1.0e-16)
        assert coeff3 == approx(0, abs=1.0e-16)

        # test cY|01> = |01>
        computer.set_state([(basis2,1.0)])
        computer.apply_gate(cY)
        coeff0 = computer.coeff(basis0)
        coeff1 = computer.coeff(basis1)
        coeff2 = computer.coeff(basis2)
        coeff3 = computer.coeff(basis3)
        assert coeff0 == approx(0, abs=1.0e-16)
        assert coeff1 == approx(0, abs=1.0e-16)
        assert coeff2 == approx(1, abs=1.0e-16)
        assert coeff3 == approx(0, abs=1.0e-16)

        # test cY|10> = i|11>
        computer.set_state([(basis1,1.0)])
        computer.apply_gate(cY)
        coeff0 = computer.coeff(basis0)
        coeff1 = computer.coeff(basis1)
        coeff2 = computer.coeff(basis2)
        coeff3 = computer.coeff(basis3)
        assert coeff0 == approx(0, abs=1.0e-16)
        assert coeff1 == approx(0, abs=1.0e-16)
        assert coeff2 == approx(0, abs=1.0e-16)
        assert coeff3.imag == approx(1, abs=1.0e-16)

        # test cY|11> = -i|10>
        computer.set_state([(basis3,1.0)])
        computer.apply_gate(cY)
        coeff0 = computer.coeff(basis0)
        coeff1 = computer.coeff(basis1)
        coeff2 = computer.coeff(basis2)
        coeff3 = computer.coeff(basis3)
        assert coeff0 == approx(0, abs=1.0e-16)
        assert coeff1.imag == approx(-1.0, abs=1.0e-16)
        assert coeff2 == approx(0, abs=1.0e-16)
        assert coeff3 == approx(0, abs=1.0e-16)

    def test_gate_ops(self):
        # test the gate operations
        gate1 = gate('X',0)
        assert gate1.nqubits() == 1
        assert gate1.gate_id() == 'X'
        assert gate1.has_parameter() == False 
        assert gate1.parameter() == None
        with pytest.raises(ValueError):
            gate1.update_parameter(1.0)

        gate2 = gate('cR',0,1,1.0)
        assert gate2.nqubits() == 2
        assert gate2.gate_id() == 'cR'
        assert gate2.has_parameter() == True
        assert gate2.parameter() == approx(1.0, abs=1.0e-16)

        gate2 = gate2.update_parameter(2.0)
        assert gate2.nqubits() == 2
        assert gate2.gate_id() == 'cR'
        assert gate2.has_parameter() == True
        assert gate2.parameter() == approx(2.0, abs=1.0e-16)




    def test_op_exp_val_1(self):
        # test direct expectation value measurement
        trial_state = Computer(4)

        trial_prep = [None]*5
        trial_prep[0] = gate('H',0,0)
        trial_prep[1] = gate('H',1,1)
        trial_prep[2] = gate('H',2,2)
        trial_prep[3] = gate('H',3,3)
        trial_prep[4] = gate('cX',0,1)

        trial_circ = Circuit()

        #prepare the circuit
        for gate_ in trial_prep:
            trial_circ.add(gate_)

        # use circuit to prepare trial state
        trial_state.apply_circuit(trial_circ)

        # gates needed for [a1^ a2] operator
        X1 = gate('X',1,1)
        X2 = gate('X',2,2)
        Y1 = gate('Y',1,1)
        Y2 = gate('Y',2,2)

        # initialize circuits to make operator
        circ1 = Circuit()
        circ1.add(X2)
        circ1.add(Y1)
        circ2 = Circuit()
        circ2.add(Y2)
        circ2.add(Y1)
        circ3 = Circuit()
        circ3.add(X2)
        circ3.add(X1)
        circ4 = Circuit()
        circ4.add(Y2)
        circ4.add(X1)

        #build the quantum operator for [a1^ a2]
        a1_dag_a2 = QubitOperator()
        a1_dag_a2.add(0.0-0.25j, circ1)
        a1_dag_a2.add(0.25, circ2)
        a1_dag_a2.add(0.25, circ3)
        a1_dag_a2.add(0.0+0.25j, circ4)

        #get direct expectatoin value
        exp = trial_state.direct_op_exp_val(a1_dag_a2)
        assert exp == approx(0.25, abs=2.0e-16)
