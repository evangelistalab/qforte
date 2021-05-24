import unittest
# import our `pybind11`-based extension module from package qforte
from qforte import qforte

# this function creates a Basis object from a string representation
def make_basis(str):
    return qforte.QuantumBasis(int(str[::-1], 2))

class GatesTests(unittest.TestCase):
    def test_X_gate(self):
        # test the Pauli X gate
        nqubits = 1
        basis0 = make_basis('0')
        basis1 = make_basis('1')
        computer = qforte.QuantumComputer(nqubits)
        X = qforte.'X',0);
        # test X|0> = |1>
        computer.apply_gate(X)
        coeff0 = computer.coeff(basis0)
        coeff1 = computer.coeff(basis1)
        self.assertAlmostEqual(coeff0, 0.0 + 0.0j)
        self.assertAlmostEqual(coeff1, 1.0 + 0.0j)
        # test X|1> = |0>
        computer.set_state([(basis1,1.0)])
        computer.apply_gate(X)
        coeff0 = computer.coeff(basis0)
        coeff1 = computer.coeff(basis1)
        self.assertAlmostEqual(coeff0, 1.0 + 0.0j)
        self.assertAlmostEqual(coeff1, 0.0 + 0.0j)


    def test_Y_gate(self):
        # test the Pauli Y gate
        nqubits = 1
        basis0 = make_basis('0')
        basis1 = make_basis('1')
        computer = qforte.QuantumComputer(nqubits)
        Y = qforte.'Y',0,0);
        # test Y|0> = i|1>
        computer.apply_gate(Y)
        coeff0 = computer.coeff(basis0)
        coeff1 = computer.coeff(basis1)
        self.assertAlmostEqual(coeff0, 0.0 + 0.0j)
        self.assertAlmostEqual(coeff1, 0.0 + 1.0j)
        # test Y|1> = -i|0>
        computer.set_state([(basis1,1.0)])
        computer.apply_gate(Y)
        coeff0 = computer.coeff(basis0)
        coeff1 = computer.coeff(basis1)
        self.assertAlmostEqual(coeff0, 0.0 - 1.0j)
        self.assertAlmostEqual(coeff1, 0.0 + 0.0j)


    def test_Z_gate(self):
        # test the Pauli Y gate
        nqubits = 1
        basis0 = make_basis('0')
        basis1 = make_basis('1')
        computer = qforte.QuantumComputer(nqubits)
        Z = qforte.'Z',0,0);
        # test Z|0> = |0>
        computer.apply_gate(Z)
        coeff0 = computer.coeff(basis0)
        coeff1 = computer.coeff(basis1)
        self.assertAlmostEqual(coeff0, 1.0 + 0.0j)
        self.assertAlmostEqual(coeff1, 0.0 + 0.0j)
        # test Z|1> = -|1>
        computer.set_state([(basis1,1.0)])
        computer.apply_gate(Z)
        coeff0 = computer.coeff(basis0)
        coeff1 = computer.coeff(basis1)
        self.assertAlmostEqual(coeff0, 0.0 + 0.0j)
        self.assertAlmostEqual(coeff1, -1.0 + 0.0j)

    def test_cX_gate(self):
        # test the cX/CNOT gate
        nqubits = 2
        basis0 = make_basis('00') # basis0:|00>
        basis1 = make_basis('01') # basis1:|10>
        basis2 = make_basis('10') # basis2:|01>
        basis3 = make_basis('11') # basis3:|11>
        computer = qforte.QuantumComputer(nqubits)
        CNOT = qforte.'CNOT',0,1);

        # test CNOT|00> = |00>
        computer.set_state([(basis0,1.0)])
        computer.apply_gate(CNOT)
        coeff0 = computer.coeff(basis0)
        coeff1 = computer.coeff(basis1)
        coeff2 = computer.coeff(basis2)
        coeff3 = computer.coeff(basis3)
        self.assertAlmostEqual(coeff0, 1.0 + 0.0j)
        self.assertAlmostEqual(coeff1, 0.0 + 0.0j)
        self.assertAlmostEqual(coeff2, 0.0 + 0.0j)
        self.assertAlmostEqual(coeff3, 0.0 + 0.0j)

        # test CNOT|10> = |11>
        computer.set_state([(basis1,1.0)])
        computer.apply_gate(CNOT)
        coeff0 = computer.coeff(basis0)
        coeff1 = computer.coeff(basis1)
        coeff2 = computer.coeff(basis2)
        coeff3 = computer.coeff(basis3)
        self.assertAlmostEqual(coeff0, 0.0 + 0.0j)
        self.assertAlmostEqual(coeff1, 0.0 + 0.0j)
        self.assertAlmostEqual(coeff2, 0.0 + 0.0j)
        self.assertAlmostEqual(coeff3, 1.0 + 0.0j)

        # test CNOT|01> = |01>
        computer.set_state([(basis2,1.0)])
        computer.apply_gate(CNOT)
        coeff0 = computer.coeff(basis0)
        coeff1 = computer.coeff(basis1)
        coeff2 = computer.coeff(basis2)
        coeff3 = computer.coeff(basis3)
        self.assertAlmostEqual(coeff0, 0.0 + 0.0j)
        self.assertAlmostEqual(coeff1, 0.0 + 0.0j)
        self.assertAlmostEqual(coeff2, 1.0 + 0.0j)
        self.assertAlmostEqual(coeff3, 0.0 + 0.0j)

        # test CNOT|11> = |10>
        computer.set_state([(basis3,1.0)])
        computer.apply_gate(CNOT)
        coeff0 = computer.coeff(basis0)
        coeff1 = computer.coeff(basis1)
        coeff2 = computer.coeff(basis2)
        coeff3 = computer.coeff(basis3)
        self.assertAlmostEqual(coeff0, 0.0 + 0.0j)
        self.assertAlmostEqual(coeff1, 1.0 + 0.0j)
        self.assertAlmostEqual(coeff2, 0.0 + 0.0j)
        self.assertAlmostEqual(coeff3, 0.0 + 0.0j)

        with self.assertRaises(ValueError) as context:
            qforte.'CNOT',0,1.0)
            self.assertTrue(')' in str(context.exception))

    def test_cY_gate(self):
        # test the cY gate
        nqubits = 2
        basis0 = make_basis('00') # basis0:|00>
        basis1 = make_basis('01') # basis1:|10>
        basis2 = make_basis('10') # basis2:|01>
        basis3 = make_basis('11') # basis3:|11>
        computer = qforte.QuantumComputer(nqubits)
        cY = qforte.'cY',0,1);

        # test cY|00> = |00>
        computer.set_state([(basis0,1.0)])
        computer.apply_gate(cY)
        coeff0 = computer.coeff(basis0)
        coeff1 = computer.coeff(basis1)
        coeff2 = computer.coeff(basis2)
        coeff3 = computer.coeff(basis3)
        self.assertAlmostEqual(coeff0, 1.0 + 0.0j)
        self.assertAlmostEqual(coeff1, 0.0 + 0.0j)
        self.assertAlmostEqual(coeff2, 0.0 + 0.0j)
        self.assertAlmostEqual(coeff3, 0.0 + 0.0j)

        # test cY|01> = |01>
        computer.set_state([(basis2,1.0)])
        computer.apply_gate(cY)
        coeff0 = computer.coeff(basis0)
        coeff1 = computer.coeff(basis1)
        coeff2 = computer.coeff(basis2)
        coeff3 = computer.coeff(basis3)
        self.assertAlmostEqual(coeff0, 0.0 + 0.0j)
        self.assertAlmostEqual(coeff1, 0.0 + 0.0j)
        self.assertAlmostEqual(coeff2, 1.0 + 0.0j)
        self.assertAlmostEqual(coeff3, 0.0 + 0.0j)

        # test cY|10> = i|11>
        computer.set_state([(basis1,1.0)])
        computer.apply_gate(cY)
        coeff0 = computer.coeff(basis0)
        coeff1 = computer.coeff(basis1)
        coeff2 = computer.coeff(basis2)
        coeff3 = computer.coeff(basis3)
        self.assertAlmostEqual(coeff0, 0.0 + 0.0j)
        self.assertAlmostEqual(coeff1, 0.0 + 0.0j)
        self.assertAlmostEqual(coeff2, 0.0 + 0.0j)
        self.assertAlmostEqual(coeff3, 0.0 + 1.0j)

        # test cY|11> = -i|10>
        computer.set_state([(basis3,1.0)])
        computer.apply_gate(cY)
        coeff0 = computer.coeff(basis0)
        coeff1 = computer.coeff(basis1)
        coeff2 = computer.coeff(basis2)
        coeff3 = computer.coeff(basis3)
        self.assertAlmostEqual(coeff0, 0.0 + 0.0j)
        self.assertAlmostEqual(coeff1, 0.0 - 1.0j)
        self.assertAlmostEqual(coeff2, 0.0 + 0.0j)
        self.assertAlmostEqual(coeff3, 0.0 + 0.0j)


    def test_computer(self):
        print('\n')
        # test that 1 - 1 = 0

        # print('\n'.join(qc.str()))
        X = qforte.'X',0,0);
        print(X)
        Y = qforte.'Y',0,0);
        print(Y)
        Z = qforte.'Z',0,0);
        print(Z)
        H = qforte.'H',0,0);
        print(H)
        R = qforte.'R',0,0,0.1);
        print(R)
        S = qforte.'S',0,0);
        print(S)
        T = qforte.'T',0,0);
        print(T)
        cX = qforte.'cX',0,1);
        print(cX)
        cY = qforte.'cY',0,1);
        print(cY)
        cZ = qforte.'cZ',0,1);
        print(cZ)
       # qcircuit = qforte.QuantumCircuit()
       # qcircuit.add_gate(qg)
       # qcircuit.add_gate(qforte.QuantumGate(qforte.QuantumGateType.Hgate,1,1));
       # print('\n'.join(qcircuit.str()))
       # self.assertEqual(qforte.subtract(1, 1), 0)

        computer = qforte.QuantumComputer(16)
       # print(repr(computer))
       # circuit = qforte.QuantumCircuit()
       # circuit.add_gate(X)
        for i in range(3000):
            computer.apply_gate(X)
            computer.apply_gate(Y)
            computer.apply_gate(Z)
            computer.apply_gate(H)
       # print(repr(computer))

    def test_op_exp_val_1(self):
        # test direct expectation value measurement
        trial_state = qforte.QuantumComputer(4)

        trial_prep = [None]*5
        trial_prep[0] = qforte.'H',0,0)
        trial_prep[1] = qforte.'H',1,1)
        trial_prep[2] = qforte.'H',2,2)
        trial_prep[3] = qforte.'H',3,3)
        trial_prep[4] = qforte.'cX',0,1)

        trial_circ = qforte.QuantumCircuit()

        #prepare the circuit
        for gate in trial_prep:
            trial_circ.add_gate(gate)

        # use circuit to prepare trial state
        trial_state.apply_circuit(trial_circ)

        # gates needed for [a1^ a2] operator
        X1 = qforte.'X',1,1)
        X2 = qforte.'X',2,2)
        Y1 = qforte.'Y',1,1)
        Y2 = qforte.'Y',2,2)

        # initialize circuits to make operator
        circ1 = qforte.QuantumCircuit()
        circ1.add_gate(X2)
        circ1.add_gate(Y1)
        circ2 = qforte.QuantumCircuit()
        circ2.add_gate(Y2)
        circ2.add_gate(Y1)
        circ3 = qforte.QuantumCircuit()
        circ3.add_gate(X2)
        circ3.add_gate(X1)
        circ4 = qforte.QuantumCircuit()
        circ4.add_gate(Y2)
        circ4.add_gate(X1)

        #build the quantum operator for [a1^ a2]
        a1_dag_a2 = qforte.QuantumOperator()
        a1_dag_a2.add_term(0.0-0.25j, circ1)
        a1_dag_a2.add_term(0.25, circ2)
        a1_dag_a2.add_term(0.25, circ3)
        a1_dag_a2.add_term(0.0+0.25j, circ4)

        #get direct expectatoin value
        exp = trial_state.direct_op_exp_val(a1_dag_a2)
        self.assertAlmostEqual(exp, 0.2499999999999999 + 0.0j)


if __name__ == '__main__':
    unittest.main()
