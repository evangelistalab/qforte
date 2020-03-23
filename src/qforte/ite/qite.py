"""
qite.py
=================================================
A module for calculating the energies of quantum-
mechanical systems using the quantum imaginary
time evolution algorithm.
"""
import qforte
from qforte.ite.qite_helpers import QITE

def qite_energy(ref,
                mol,
                beta,
                db,
                verbose = False,
                expansion_type = 'qbGSD',
                state_prep = 'single_reference',
                trotter_number = 1,
                fast = True,
                sparseSb = True):

    """Uses the QITE algorithm to build the ground-state and calculate the
        energy.

        Arguments
        ---------
        ref : list
            The set of 1s and 0s indicating the initial quantum state.

        mol : Molecule
            The Molecule object to use in ADAPT-VQE. Specifies the Hamiltonian.

        beta : float
            The total imaginary evolution time.

        db : float
            The imaginary time step.

        verbose : bool
            Whether or not to print additional details like the b vectors and the
            quantum computer state at each evolution time beta_k = k*db.

        expansion_type : string
            The expansion basis for A.
            Can any one of the follwing,
                (1) 'sqGSD' : a basis of generalized second-quantized singles and
                doubels operators (still only for singlet states).
                (2) 'qbGSD' : same as above but does not cancel imaginary terms
                when converting to a pauli operator representaion.
                (3) 'complete_qubit' : all possible configurations of pauli operators,
                will result in a basis of size 4^nqubits.

        state_prep : string
            How to use the reference to construct the initial state preperation
            circuit. Currently only supports 'single_reference'.

        trott_number : int
            The Trotter number for the calculation
            (exact in the infinte limit).

        fast : bool
            Whether or not to use a faster version of the algorithm that bypasses
            measurment (unphysical for quantum computer).

        sparseSB : bool
            Whether or not to build a S and bl using only non-zero elements.

    """

    print('\n-----------------------------------------------------')
    print('     Quantum Imaginary Time Evolution Algorithm   ')
    print('-----------------------------------------------------')

    n_qubits = len(ref)
    # TODO: move the below function 'ref_to_basis_idx' to helper folder
    init_basis_idx = qforte.qkd.qk_helpers.ref_to_basis_idx(ref)
    init_basis = qforte.QuantumBasis(init_basis_idx)

    print('\n\n                 ==> QITE options <==')
    print('-----------------------------------------------------------')
    print('Initial reference state:                 ',  init_basis.str(n_qubits))
    print('State preparation method:                ',  state_prep)
    print('Expansion type:                          ',  expansion_type)
    print('Total imaginary evolution time (beta):   ',  beta)
    print('Imaginary time step (db):                ',  db)
    print('Trotter number (m):                      ',  trotter_number)
    print('Use fast version of algorithm:           ',  str(fast))
    print('Use sparse tensors to solve Sx = b:      ',  str(sparseSb))
    print('Number of measurements per term:         ',  'infinite')

    myQITE = QITE(ref,
                  mol.get_hamiltonian(),
                  mol.get_sq_hamiltonian(),
                  beta,
                  db,
                  verbose = verbose,
                  expansion_type = expansion_type,
                  state_prep = state_prep,
                  trotter_number = trotter_number,
                  fast = fast,
                  sparseSb = sparseSb)

    myQITE.evolve()
    Eqite = myQITE._Ekb[-1]

    print('\n\n                        ==> QITE summary <==')
    print('---------------------------------------------------------------')
    print('Final QITE Energy:                        ', round(Eqite, 10))

    return Eqite
