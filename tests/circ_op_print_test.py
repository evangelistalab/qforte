from io import StringIO
import unittest
from unittest import TestCase
from unittest.mock import patch
import qforte as qf
from qforte import *

class Circ_Op_PrintTest(TestCase):
    def test_circ_op_print(self):

        # initialize empty circuit
        circ1 = qf.Circuit()
        # add (Z1 H2 Y4 X4) Pauli string
        # note: the rightmost gate is applied first
        circ1.add(qf.gate('X', 4))
        circ1.add(qf.gate('Y', 4))
        circ1.add(qf.gate('H', 2))
        circ1.add(qf.gate('Z', 1))

        # test built-in print function
        out = str(circ1)
        self.assertEqual(out, '[Z1 H2 Y4 X4]')

        # test smart_print function
        with patch('sys.stdout', new = StringIO()) as fake_out:
            smart_print(circ1)
            self.assertEqual(fake_out.getvalue(), '\n Quantum circuit:\n(Z1 H2 Y4 X4) |Ψ>\n')

        # initialize empty circuit
        circ2 = qf.Circuit()
        # add (H1 Y2 S3 I4) Pauli string
        # note: the rightmost gate is applied first
        circ2.add(qf.gate('I', 4))
        circ2.add(qf.gate('S', 3))
        circ2.add(qf.gate('Y', 2))
        circ2.add(qf.gate('H', 1))

        # initialize empty circuit
        circ3 = qf.Circuit()
        # add (X1 Y2 H3 Z4) Pauli string
        # note: the rightmost gate is applied first
        circ3.add(qf.gate('Z', 4))
        circ3.add(qf.gate('H', 3))
        circ3.add(qf.gate('Y', 2))
        circ3.add(qf.gate('X', 1))

        # initialize empty QubitOperator
        q_op = qf.QubitOperator()

        # add u1*circ1 + u2*circ2 + u3*circ3
        u1 = 0.5+0.1j
        u2 = -0.5j
        u3 = +0.3

        q_op.add(u1, circ1)
        q_op.add(u2, circ2)
        q_op.add(u3, circ3)

        # test built-in print function
        out = str(q_op)
        self.assertEqual(out, '+0.500000 +0.100000i[Z1 H2 Y4 X4]\n'\
                         '-0.500000j[H1 Y2 S3 I4]\n'\
                         '+0.300000[X1 Y2 H3 Z4]')

        # test smart_print function
        with patch('sys.stdout', new = StringIO()) as fake_out:
            smart_print(q_op)
            self.assertEqual(fake_out.getvalue(), '\n Quantum operator:\n(0.5+0.1j) (Z1 H2 Y4 X4) |Ψ>\n'\
                             '+ (-0-0.5j) (H1 Y2 S3 I4) |Ψ>\n'\
                             '+ (0.3+0j) (X1 Y2 H3 Z4) |Ψ>\n')
if __name__ == '__main__':
    unittest.main()
