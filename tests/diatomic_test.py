import unittest
import qforte
from openfermion.hamiltonians import MolecularData
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermionpsi4 import run_psi4
import numpy

# Set parameters to make a simple molecule.
class HelperTests(unittest.TestCase):
    def test_h2(self):
        diatomic_bond_length = .7414
        geometry = [('H', (0., 0., 0.)), ('H', (0., 0., diatomic_bond_length))]
        basis = 'sto-3g'
        multiplicity = 1
        charge = 0
        description = str(diatomic_bond_length)
        
        #Generate molecular data and run psi4 calculation
        molecule = MolecularData(geometry, basis, multiplicity,
                                 charge, description)
        
        molecule = run_psi4(molecule,
                                run_scf=1,
                                run_mp2=0,
                                run_cisd=0,
                                run_ccsd=0,
                                run_fci=0)
        
        print('\nPsi4 calculations:')
        print('\nAt bond length of {} angstrom, molecular hydrogen has:'.format(
                diatomic_bond_length))
        print('Hartree-Fock energy of {} Hartree.'.format(molecule.hf_energy))
        
        #populate data and integrals
        molecule.load()
        
        # Get the Hamiltonian in an active space.
        active_space_start = 0
        active_space_stop = 1
        
        molecular_hamiltonian = molecule.get_molecular_hamiltonian(
            occupied_indices=range(active_space_start),
            active_indices=range(active_space_start, active_space_stop))
        
        # Map operator to fermions and qubits.
        fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)
        qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
        #qubit_hamiltonian.compress()
        print('The Jordan-Wigner Molecular Hamiltonian in canonical basis:\n{}'.format(qubit_hamiltonian))

        print('\nBuild Qforte Hamiltonian and generator to calculate the expectation value')
        qforte_hamiltonian = qforte.build_from_openfermion(qubit_hamiltonian)
        print('\nThe Molecular Hamiltonian in canonical basis:')
        qforte.smart_print(qforte_hamiltonian)
        
        trial_state = qforte.QuantumComputer(4)
        
        #Build a Hartree-Fock trial state
        circ = qforte.QuantumCircuit()
        circ.add_gate(qforte.make_gate('X', 0, 0))
        circ.add_gate(qforte.make_gate('X', 1, 1))
        trial_state.apply_circuit(circ)
        qforte.smart_print(trial_state)
        
        exp = trial_state.direct_op_exp_val(qforte_hamiltonian)
        print(exp)
        self.assertAlmostEqual(exp, -1.1166843870661929 + 0.0j)
    
if __name__ == '__main__':
    unittest.main()
