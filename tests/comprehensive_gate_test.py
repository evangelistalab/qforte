import unittest
# import our `pybind11`-based extension module from package qforte
from qforte import qforte
import numpy as np

num_qubits = 5
prep_circ = qforte.Circuit()
ct_lst = [(4,3), (4,2), (4,1), (4,0), (3,2), (3,1), (3,0), (2,1), (2,0), (1,0)]

for i in range(num_qubits):
    prep_circ.add(qforte.gate('H',i, i))

for i in range(num_qubits):
    prep_circ.add(qforte.gate('cR',i, i+1, 1.116 / (i+1.0)))

def generic_test_circ_vec_builder(qb_list, id):
    circ_vec_tc = [qforte.Circuit() for i in range(len(qb_list))]
    circ_vec_ct = [qforte.Circuit() for i in range(len(qb_list))]
    for i, pair in enumerate(ct_lst):
        t = pair[0]
        c = pair[1]
        if(id == 'cR'):
            circ_vec_ct[i].add(qforte.gate(id, t, c, 3.17*t*c))
            circ_vec_tc[i].add(qforte.gate(id, c, t, 1.41*t*c))

        else:
            circ_vec_ct[i].add(qforte.gate(id, t, c))
            circ_vec_tc[i].add(qforte.gate(id, c, t))

    return circ_vec_tc, circ_vec_ct

def circuit_tester(prep, test_circ):
    for gate in test_circ.gates():
        id = gate.gate_id()
        target = gate.target()
        control = gate.control()

        num_qubits = 5

        qc1 = qforte.Computer(num_qubits)
        qc2 = qforte.Computer(num_qubits)

        qc1.apply_circuit_safe(prep)
        qc2.apply_circuit_safe(prep)

        qc1.apply_gate_safe(gate)
        qc2.apply_gate(gate)

        C1 = qc1.get_coeff_vec()
        C2 = qc2.get_coeff_vec()

        diff_vec = [ (C1[i] - C2[i])*np.conj(C1[i] - C2[i]) for i in range(len(C1))]
        diff_norm = np.sum(diff_vec)

        if(np.sum(diff_vec) != (0.0 + 0.0j)):
            print('|C - C_safe|F^2   control   target   id')
            print('----------------------------------------')
            print(diff_norm,'              ',control,'       ',target,'      ',id)

        return diff_norm

class ComprehensiveGatesTests(unittest.TestCase):
    def test_comp_cX_gates(self):
        print('\nSTART test_comp_cX_gates\n')
        id = 'cX'
        circ_tc, circ_ct = generic_test_circ_vec_builder(ct_lst, id)

        print('\n-------------------')
        print('Testing ' + id + ' circuits')
        print('-------------------')

        for circ in circ_ct:
            ct_val = circuit_tester(prep_circ, circ)
            self.assertAlmostEqual(ct_val, 0.0 + 0.0j)

        for circ in circ_tc:
            tc_val = circuit_tester(prep_circ, circ)
            self.assertAlmostEqual(tc_val, 0.0 + 0.0j)

    def test_comp_cY_gates(self):
        print('\nSTART test_comp_cY_gates\n')
        id = 'cY'
        circ_tc, circ_ct = generic_test_circ_vec_builder(ct_lst, id)

        print('\n-------------------')
        print('Testing ' + id + ' circuits')
        print('-------------------')

        for circ in circ_ct:
            ct_val = circuit_tester(prep_circ, circ)
            self.assertAlmostEqual(ct_val, 0.0 + 0.0j)

        for circ in circ_tc:
            tc_val = circuit_tester(prep_circ, circ)
            self.assertAlmostEqual(tc_val, 0.0 + 0.0j)

    def test_comp_cZ_gates(self):
        print('\nSTART test_comp_cZ_gates\n')
        id = 'cZ'
        circ_tc, circ_ct = generic_test_circ_vec_builder(ct_lst, id)

        print('\n-------------------')
        print('Testing ' + id + ' circuits')
        print('-------------------')

        for circ in circ_ct:
            ct_val = circuit_tester(prep_circ, circ)
            self.assertAlmostEqual(ct_val, 0.0 + 0.0j)

        for circ in circ_tc:
            tc_val = circuit_tester(prep_circ, circ)
            self.assertAlmostEqual(tc_val, 0.0 + 0.0j)

    def test_comp_cR_gates(self):
        print('\nSTART test_comp_cR_gates\n')
        id = 'cR'
        circ_tc, circ_ct = generic_test_circ_vec_builder(ct_lst, id)

        print('\n-------------------')
        print('Testing ' + id + ' circuits')
        print('-------------------')

        for circ in circ_ct:
            ct_val = circuit_tester(prep_circ, circ)
            self.assertAlmostEqual(ct_val, 0.0 + 0.0j)

        for circ in circ_tc:
            tc_val = circuit_tester(prep_circ, circ)
            self.assertAlmostEqual(tc_val, 0.0 + 0.0j)

    def test_comp_cV_gates(self):
        print('\nSTART test_comp_cV_gates\n')
        id = 'cV'
        circ_tc, circ_ct = generic_test_circ_vec_builder(ct_lst, id)

        print('\n-------------------')
        print('Testing ' + id + ' circuits')
        print('-------------------')

        for circ in circ_ct:
            ct_val = circuit_tester(prep_circ, circ)
            self.assertAlmostEqual(ct_val, 0.0 + 0.0j)

        for circ in circ_tc:
            tc_val = circuit_tester(prep_circ, circ)
            self.assertAlmostEqual(tc_val, 0.0 + 0.0j)

if __name__ == '__main__':
    unittest.main()
