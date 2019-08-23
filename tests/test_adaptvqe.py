import qforte
import unittest

class AdaptVQETests(unittest.TestCase):
    def test_h2_1(self):
        r = 0.7414
        geom = [('H', (0, 0, 0)), ('H', (0, 0, r))]
        psi4_setup = {'description': F'H2-{r}',
                  'run_scf': 1,
                  'run_fci': 1}
        mol = qforte.runPsi4(geom, psi4_setup)
        adapt_instance = qforte.ADAPT_VQE(mol)
        adapt_instance.build_operator_pool()
        adapt_instance.iterating_cycle(print_details=False)
        adapt_energy = adapt_instance.energy[-1]

        self.assertAlmostEqual(adapt_energy, mol.fci_energy, places=10)

    def test_h2_2(self):
        r = 1.5
        geom = [('H', (0, 0, 0)), ('H', (0, 0, r))]
        psi4_setup = {'description': F'H2-{r}',
                      'run_scf': 1,
                      'run_fci': 1}
        mol = qforte.runPsi4(geom, psi4_setup)
        adapt_instance = qforte.ADAPT_VQE(mol)
        adapt_instance.build_operator_pool()
        adapt_instance.iterating_cycle(print_details=False)
        adapt_energy = adapt_instance.energy[-1]

        self.assertAlmostEqual(adapt_energy, mol.fci_energy, places=10)

if __name__ == '__main__':
    unittest.main()


