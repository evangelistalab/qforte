import unittest
from qforte import qforte as qf

class BasisTest(unittest.TestCase):
    def test_str(self):
        print('\nSTART test_str\n')
        self.assertEqual(str(qf.QubitBasis(0)), "|0000000000000000000000000000000000000000000000000000000000000000>")
        self.assertEqual(str(qf.QubitBasis(5)), "|1010000000000000000000000000000000000000000000000000000000000000>")

if __name__ == '__main__':
    unittest.main()
