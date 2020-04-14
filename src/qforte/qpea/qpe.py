"""
qpe.py
=================================================
A module for calculating the energies of quantum-
mechanical systems using the quatum phase
estimation algorithm.
"""

import qforte
from qforte.qpea import qpe_helpers
from qforte.qpea.qpe_helpers import *

import numpy as np
from scipy import stats

def qpe_energy(ref, mol,
                t = 1.0,
                nruns = 100,
                state_prep = 'single_reference',
                success_prob = 0.5,
                num_precise_bits = 10,
                trotter_number=1,
                fast=True,
                return_phases=False):
    """Uses the canonical quantum phase estimation algorithm to determine
    a binrary approximation of the phase corresponding the the ground state
    eigenvalue. The phase e^-iEt = e^-2pi * binary(phi) * t, where
    binary(phi) = [0.b1b2b3..bn], and b1...bn are 0 or 1 determined by measurement
    of the ancilla register in the computational basis.

    See text "Quantum Computation and Quantum Infromation" by Michael A. Neilsen
    and Issac L. Chuang page 221 for deatals.

        Arguments
        ---------

        ref : list
            The initial reference state given as a list of 1's and 0's
            (e.g. the Hartree-Fock state).

        mol : Molecule
            The Molecule object to which the Hamiltonain used for the phase
            estimation pertains.

        t : float
            The total evolution time.

        nruns : int
            The number of times to repeat the algorithm to produce a phase.

        state_prep : string
            The desired approach for inital state preparation. Specifying
            'single_reference' will build a circuit which will reproduce the
            porvieded ref state.

        success_prob = float
            A number between 0.0 and 1.0 indicatoin the success probablity of
            measuring the eivenvalue, to obtain a higer success probablity more
            ancilla qubits (and thus more time steps) will be used.

        num_precise_bits : int
            The number of digits of binary precision to target. For example,
            using n bit precision, one can acheive errors in the phase on the
            order of 2^-n.

        trotter_number : int
            The trotter number (m) to use for the decompostion. Exponentiation
            is exact in the m --> infinity limit.

        fast : bool
            Whether or not to use a faster version of the algorithm that bypasses
            measurment (unphysical for quantum computer).

        return_phases : bool
            Whether or not to return a list of containing all measured phases


        Returns
        -------

        final_energy : float
            The final energy value calculated from the measured binary phase.
            E = - 2 pi [0.b1b2b3...bn] / t,
            where [0.b1b2b3...bn] = b1 / 2 + b2 / 4 + ... + bn / 2^n

        phases : list
            The decimeal representatoin of the measured phases for all nruns.
    """

    print('\n-----------------------------------------------------')
    print('       Quantum Phase Estimation Algorithm   ')
    print('-----------------------------------------------------')

    n_state_qubits = len(ref)
    eps = 1 - success_prob
    n_ancilla = num_precise_bits + int(np.log2(2 + (1.0/eps)))
    n_tot_qubits = n_state_qubits + n_ancilla

    init_basis_idx = qforte.qkd.qk_helpers.ref_to_basis_idx(ref)
    init_basis = qforte.QuantumBasis(init_basis_idx)

    print('\n\n                 ==> QPE options <==')
    print('-----------------------------------------------------------')
    print('Reference state:                         ',  init_basis.str(n_state_qubits))
    print('State preparation method:                ',  state_prep)
    print('Target success probability:              ',  success_prob)
    print('Number of precise bits for phase:        ',  num_precise_bits)
    print('Number of time steps:                    ',  n_ancilla)
    print('Evolution time (t):                      ',  t)
    print('Trotter number (m):                      ',  trotter_number)
    print('Number of algorithm executions:          ',  nruns)
    print('Use fast version of algorithm:           ',  str(fast))
    #TODO: enable finite measurement per term (Nick)
    print('Number of measurements per term:         ',  'infinite')

    abegin = n_state_qubits
    aend = n_tot_qubits - 1

    # build hadamard circ
    Uhad = get_Uhad(abegin, aend)

    # build preparation circuit
    Uprep = get_Uprep(ref, state_prep)

    # build controll e^-iHt circuit
    Udyn = get_dynamics_circ(mol.get_hamiltonian(),
                             trotter_number,
                             abegin,
                             n_ancilla,
                             t=t)

    # build reverse QFT
    revQFTcirc = qft_circuit(abegin, aend, 'reverse')

    # build QPEcirc
    QPEcirc = qforte.QuantumCircuit()
    QPEcirc.add_circuit(Uprep)
    QPEcirc.add_circuit(Uhad)
    QPEcirc.add_circuit(Udyn)
    QPEcirc.add_circuit(revQFTcirc)

    computer = qforte.QuantumComputer(n_tot_qubits)
    computer.apply_circuit(QPEcirc)

    if fast:
        z_readouts = computer.measure_z_readouts_fast(abegin, aend, nruns)

    else:
        Zcirc = get_z_circuit(abegin, aend)
        z_readouts = computer.measure_readouts(Zcirc, nruns)

    final_energy = 0.0
    phases = []
    for readout in z_readouts:
        val = 0.0
        i = 1
        for z in readout:
            val += z / (2**i)
            i += 1
        phases.append(val)

    # find final binary string:
    final_readout = []
    final_readout_aves = []
    for i in range(n_ancilla):
        iave = 0.0
        for readout in z_readouts:
            iave += readout[i]
        iave /= nruns
        final_readout_aves.append(iave)
        if (iave > (1.0/2)):
            final_readout.append(1)
        else:
            final_readout.append(0)

    final_phase = 0.0
    counter = 0
    for i, z in enumerate(final_readout):
            final_phase += z / (2**(i+1))

    final_energy = -2 * np.pi * final_phase / t

    ave_energy = -1.0*np.mean(np.asarray(phases))
    res = stats.mode(np.asarray(phases))
    mode_phase = res.mode[0]
    mode_energy = -2 * np.pi * mode_phase / t

    print('\n           ==> QPE readout averages <==')
    print('------------------------------------------------')
    for i, ave in enumerate(final_readout_aves):
        print('  bit ', i,  ': ', ave)
    print('\n  Final bit readout: ', final_readout)

    print('\n\n                        ==> QPE summary <==')
    print('---------------------------------------------------------------')
    print('FCI Energy:                              ', round(mol.get_fci_energy()[()], 10))
    print('Final QPE Energy:                        ', round(final_energy, 10))
    print('Mode QPE Energy:                         ', round(mode_energy, 10))
    print('Final QPE phase:                          ', round(final_phase, 10))
    print('Mode QPE phase:                           ', round(mode_phase, 10))

    if return_phases:
        return final_energy, phases
    else:
        return final_energy
