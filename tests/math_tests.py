import unittest
# import our `pybind11`-based extension module from package qforte 
from qforte import qforte

class MainTest(unittest.TestCase):
    def test_add(self):
        # test that 1 + 1 = 2
        self.assertEqual(qforte.add(1, 1), 2)

    def test_subtract(self):
        # test that 1 - 1 = 0
        self.assertEqual(qforte.subtract(1, 1), 0)

if __name__ == '__main__':
    unittest.main()
