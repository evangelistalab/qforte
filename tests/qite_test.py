import unittest
import numpy as np
from qforte import qforte
from qforte.ite.qite_helpers import QITE

class QITETests(unittest.TestCase):
    def test_H2(self):
        print('\n'),
        # The FCI energy for H2 at 1.5 Angstrom in a sto-3g basis
        E_fci = -0.9981493534

        coef_vec = [-0.4917857774144603,
                    0.09345649662931771,
                    0.09345649662931771,
                    -0.0356448161226769,
                    -0.0356448161226769,
                    0.1381758457453024,
                    0.05738398402634884,
                    -0.0573839840263488,
                    -0.0573839840263488,
                    0.05738398402634884,
                    0.08253705485911705,
                    0.13992103888546592,
                    0.13992103888546592,
                    0.08253705485911705,
                    0.1458551902800438]

        circ_vec = [
        qforte.QuantumCircuit( ),
        qforte.build_circuit( 'Z_0' ),
        qforte.build_circuit( 'Z_1' ),
        qforte.build_circuit( 'Z_2' ),
        qforte.build_circuit( 'Z_3' ),
        qforte.build_circuit( 'Z_0   Z_1' ),
        qforte.build_circuit( 'Y_0   X_1   X_2   Y_3' ),
        qforte.build_circuit( 'X_0   X_1   Y_2   Y_3' ),
        qforte.build_circuit( 'Y_0   Y_1   X_2   X_3' ),
        qforte.build_circuit( 'X_0   Y_1   Y_2   X_3' ),
        qforte.build_circuit( 'Z_0   Z_2' ),
        qforte.build_circuit( 'Z_0   Z_3' ),
        qforte.build_circuit( 'Z_1   Z_2' ),
        qforte.build_circuit( 'Z_1   Z_3' ),
        qforte.build_circuit( 'Z_2   Z_3' )]

        H2_qubit_hamiltonian = qforte.QuantumOperator()
        for i in range(len(circ_vec)):
            H2_qubit_hamiltonian.add_term(coef_vec[i], circ_vec[i])

        H2_sq_hamiltonian = [
        [(), 0.3527848071133334],
        [(0, 0), -0.9081808722384057],
        [(1, 1), -0.9081808722384057],
        [(2, 2), -0.6653369358038996],
        [(3, 3), -0.6653369358038996],
        [(1, 0, 1, 0), -0.5527033829812091],
        [(1, 0, 3, 2), -0.22953593610539536],
        [(2, 0, 2, 0), -0.3301482194364681],
        [(3, 0, 2, 1), 0.22953593610539527],
        [(3, 0, 3, 0), -0.5596841555418633],
        [(2, 1, 3, 0), 0.22953593610539527],
        [(2, 1, 2, 1), -0.5596841555418633],
        [(3, 1, 3, 1), -0.3301482194364681],
        [(3, 2, 1, 0), -0.22953593610539522],
        [(3, 2, 3, 2), -0.5834207611201749]
        ]

        ref = [1,1,0,0]

        print('\nBegin QITE test for H2')
        print('-------------------------')

        myQITE = QITE(ref,
                  H2_qubit_hamiltonian,
                  H2_sq_hamiltonian,
                  8.0,
                  0.2,
                  verbose = False,
                  expansion_type = 'qbGSD',
                  state_prep = 'single_reference',
                  trotter_number = 1,
                  fast = True,
                  sparseSb = True)

        myQITE.evolve()
        final_energy = myQITE._Ekb[-1]
        print('\nEqite: ', round(final_energy, 10))
        print('Efci:  ', round(E_fci,10))
        self.assertLess(np.abs(final_energy-E_fci), 1.0e-6)


if __name__ == '__main__':
    unittest.main()
