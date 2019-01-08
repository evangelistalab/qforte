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
        X = qforte.make_gate('X',0,0);
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
        Y = qforte.make_gate('Y',0,0);
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
        Z = qforte.make_gate('Z',0,0);
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


    def test_computer(self):
        # test that 1 - 1 = 0

#        print('\n'.join(qc.str()))
        X = qforte.make_gate('X',0,0);
        print(X)
        Y = qforte.make_gate('Y',0,0);
        print(Y)
        Z = qforte.make_gate('Z',0,0);
        print(Z)
        H = qforte.make_gate('H',0,0);
        print(H)
        R = qforte.make_gate('R',0,0,0.1);
        print(R)
        S = qforte.make_gate('S',0,0);
        print(S)
        T = qforte.make_gate('T',0,0);
        print(T)
        cX = qforte.make_gate('cX',0,1);
        print(cX)
        cY = qforte.make_gate('cY',0,1);
        print(cY)
        cZ = qforte.make_gate('cZ',0,1);
        print(cZ)
#        qcircuit = qforte.QuantumCircuit()
#        qcircuit.add_gate(qg)
#        qcircuit.add_gate(qforte.QuantumGate(qforte.QuantumGateType.Hgate,1,1));
#        print('\n'.join(qcircuit.str()))
#        self.assertEqual(qforte.subtract(1, 1), 0)

        computer = qforte.QuantumComputer(16)
#        print(repr(computer))
#        circuit = qforte.QuantumCircuit()
#        circuit.add_gate(X)
        for i in range(3000):
            computer.apply_gate(X)
            computer.apply_gate(Y)
            computer.apply_gate(Z)
            computer.apply_gate(H)
#        print(repr(computer))

if __name__ == '__main__':
    unittest.main()
