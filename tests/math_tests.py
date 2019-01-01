import unittest
# import our `pybind11`-based extension module from package qforte 
from qforte import qforte

def make_basis(str):
    return qforte.Basis(int(str[::-1], 2))

class MainTest(unittest.TestCase):
    def test_X_gate(self):
        # test that 1 + 1 = 2
        nqubits = 3
        computer = qforte.QuantumComputer(nqubits)
        X = qforte.make_gate('X',0,0);
        computer.apply_gate(X)
        print(make_basis('100').str(nqubits))
        basis100 = make_basis('100')
        coeff100 = computer.coeff(basis100)
        self.assertEqual(coeff100, 1.0 + 0.0j)

#    def test_subtract(self):
#        # test that 1 - 1 = 0
#        self.assertEqual(qforte.subtract(1, 1), 0)

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

        computer = qforte.QuantumComputer(3)
        print(repr(computer))
#        circuit = qforte.QuantumCircuit()
#        circuit.add_gate(X)
        computer.apply_gate(H)
        computer.apply_gate(H)
        print(repr(computer))

if __name__ == '__main__':
    unittest.main()
