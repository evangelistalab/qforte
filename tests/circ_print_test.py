from io import StringIO
import unittest
from unittest import TestCase
from unittest.mock import patch
import qforte as qf
from qforte import *

class CircPrintTest(TestCase):
    def test_gate_print(self):

        # initialize empty circuit
        circ = qf.Circuit()
        # add (Z1 H2 Y4 X4) Pauli string
        # note: the rightmost gate is applied first
        circ.add(qf.gate('X', 4))
        circ.add(qf.gate('Y', 4))
        circ.add(qf.gate('H', 2))
        circ.add(qf.gate('Z', 1))

        # test built-in print function
        with patch('sys.stdout', new = StringIO()) as fake_out:
            print(circ)
            self.assertEqual(fake_out.getvalue(), '[Z1 H2 Y4 X4]\n')

        # test smart_print function
        with patch('sys.stdout', new = StringIO()) as fake_out:
            smart_print(circ)
            self.assertEqual(fake_out.getvalue(), '\n Quantum circuit:\n( Z1 H2 Y4 X4 ) |Ψ>\n')

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
