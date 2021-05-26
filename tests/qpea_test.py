import unittest
from qforte import qforte
from qforte.qpea.qpe import QPE
from qforte.system.molecular_info import Molecule

class QPETests(unittest.TestCase):
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
        qforte.Circuit( ),
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
            H2_qubit_hamiltonian.add(coef_vec[i], circ_vec[i])

        ref = [1,1,0,0]

        print('\nBegin QPE test for H2')
        print('----------------------')

        # make test with algorithm class
        mol = Molecule()
        mol.set_hamiltonian(H2_qubit_hamiltonian)

        alg = QPE(mol, reference=ref, trotter_number=2)
        alg.run(t = 0.4,
                nruns = 100,
                success_prob = 0.5,
                num_precise_bits = 8)

        Egs = alg.get_gs_energy()
        self.assertLess(abs(Egs-E_fci), 1.1e-3)


if __name__ == '__main__':
    unittest.main()
