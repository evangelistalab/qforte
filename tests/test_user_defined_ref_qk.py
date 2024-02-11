from pytest import approx
from qforte import system_factory, MRSQK

class TestRefSpaceMRSQK:
    def test_H4_Root1_user_defined_ref_space(self):
        print('\n')
        # FCI Root 1 singlet energy for H4 at 1.5 angstrom in sto-6g basis
        e_fci_root1 = -1.839569968502

        Rhh = 1.5
        mol = system_factory(system_type = 'molecule',
                build_type = 'psi4',
                basis = 'sto-6g',
                mol_geometry = [('H', (0, 0, -3*Rhh/2)),
                                ('H', (0, 0, -Rhh/2)),
                                ('H', (0, 0, Rhh/2)),
                                ('H', (0, 0, 3*Rhh/2))],
                symmetry = 'd2h',
                multiplicity = 1, # Only singlets will work with QForte
                charge = 0,
                num_frozen_docc = 0,
                num_frozen_uocc = 0,
                run_mp2=0,
                run_ccsd=0,
                run_cisd=0,
                run_fci=0)

        refs_user_defined = [
            [(1.0, [1,1,0,0,1,1,0,0]), ], # 22 00
            [(1.0, [0,0,1,1,1,1,0,0]), ], # 02 20
            [(1.0, [1,1,0,0,0,0,1,1]), ], # 20 02
        ]
        target_root = 1

        alg = MRSQK(mol, trotter_number=100, trotter_order=1, )
        alg.run(d = 3,  # num of references
                s = 3,  # num of time steps per ref
                mr_dt = 0.5,  
                target_root = target_root, 
                reference_generator = 'user',
                refs_user_defined = refs_user_defined,
                )
            
        E1 = alg.get_ts_energy()

        assert E1 == approx(e_fci_root1, abs=1e-5)

