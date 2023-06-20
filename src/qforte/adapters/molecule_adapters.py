"""
A class for building molecular object adapters. Adapters for various approaches to build
the molecular info and properties (hamiltonian, rdms, etc...).
"""
import operator
import numpy as np
from abc import ABC, abstractmethod

import qforte

from qforte.system.molecular_info import Molecule
from qforte.utils import transforms as tf


import json

try:
    import psi4
    use_psi4 = True
except:
    use_psi4 = False


def create_psi_mol(**kwargs):
    """Builds a qforte Molecule object directly from a psi4 calculation.

    Returns
    -------
    Molecule
        The qforte Molecule object which holds the molecular information.
    """

    kwargs.setdefault('symmetry', 'c1')
    kwargs.setdefault('charge', 0)
    kwargs.setdefault('multiplicity', 1)

    mol_geometry = kwargs['mol_geometry']
    basis = kwargs['basis']
    multiplicity = kwargs['multiplicity']
    charge = kwargs['charge']

    qforte_mol = Molecule(mol_geometry = mol_geometry,
                               basis = basis,
                               multiplicity = multiplicity,
                               charge = charge)

    if not use_psi4:
        raise ImportError("Psi4 was not imported correctely.")

    # By default, the number of frozen orbitals is set to zero
    kwargs.setdefault('num_frozen_docc', 0)
    kwargs.setdefault('num_frozen_uocc', 0)

    # run_scf is not read, because we always run SCF to get a wavefunction object.
    kwargs.setdefault('run_mp2', False)
    kwargs.setdefault('run_ccsd', False)
    kwargs.setdefault('run_cisd', False)
    kwargs.setdefault('run_fci', False)

    # Setup psi4 calculation(s)
    psi4.set_memory('2 GB')
    psi4.core.set_output_file(kwargs['filename']+'.out', False)

    p4_geom_str =  f"{int(charge)}  {int(multiplicity)}"
    for geom_line in mol_geometry:
        p4_geom_str += f"\n{geom_line[0]}  {geom_line[1][0]}  {geom_line[1][1]}  {geom_line[1][2]}"
    p4_geom_str += f"\nsymmetry {kwargs['symmetry']}"
    p4_geom_str += f"\nunits angstrom"

    print(' ==> Psi4 geometry <==')
    print('-------------------------')
    print(p4_geom_str)

    p4_mol = psi4.geometry(p4_geom_str)

    scf_ref_type = "rhf" if multiplicity == 1 else "rohf"

    psi4.set_options({'basis': basis,
              'scf_type': 'pk',
              'reference' : scf_ref_type,
              'e_convergence': 1e-8,
              'd_convergence': 1e-8,
              'ci_maxiter': 100,
              'num_frozen_docc' : kwargs['num_frozen_docc'],
              'num_frozen_uocc' : kwargs['num_frozen_uocc'],
              'mp2_type': "conv"})

    # run psi4 caclulation
    p4_Escf, p4_wfn = psi4.energy('SCF', return_wfn=True)

    # Run additional computations requested by the user
    if kwargs['run_mp2']:
        qforte_mol.mp2_energy = psi4.energy('MP2')

    if kwargs['run_ccsd']:
        qforte_mol.ccsd_energy = psi4.energy('CCSD')

    if kwargs['run_cisd']:
        qforte_mol.cisd_energy = psi4.energy('CISD')

    if kwargs['run_fci']:
        if kwargs['num_frozen_uocc'] == 0:
            qforte_mol.fci_energy = psi4.energy('FCI')
        else:
            print('\nWARNING: Skipping FCI computation due to a Psi4 bug related to FCI with frozen virtuals.\n')

    # Get integrals using MintsHelper.
    mints = psi4.core.MintsHelper(p4_wfn.basisset())

    C = p4_wfn.Ca_subset("AO", "ALL")

    scalars = p4_wfn.scalar_variables()

    p4_Enuc_ref = scalars["NUCLEAR REPULSION ENERGY"]

    # Do MO integral transformation
    mo_teis = np.asarray(mints.mo_eri(C, C, C, C))
    mo_oeis = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
    mo_oeis = np.einsum('uj,vi,uv', C, C, mo_oeis)

    nmo = np.shape(mo_oeis)[0]
    nalpha = p4_wfn.nalpha()
    nbeta = p4_wfn.nbeta()
    nel = nalpha + nbeta
    frozen_core = p4_wfn.frzcpi().sum()
    frozen_virtual = p4_wfn.frzvpi().sum()

    # Get symmetry information
    orbitals = []
    for irrep, block in enumerate(p4_wfn.epsilon_a_subset("MO", "ACTIVE").nph):
        for orbital in block:
            orbitals.append([orbital, irrep])

    orbitals.sort()
    hf_orbital_energies = []
    orb_irreps_to_int = []
    for row in orbitals:
        hf_orbital_energies.append(row[0])
        orb_irreps_to_int.append(row[1])
    del orbitals

    point_group = p4_mol.symmetry_from_input().lower()
    irreps = qforte.irreps_of_point_groups(point_group)
    orb_irreps = [irreps[i] for i in orb_irreps_to_int]

    # If frozen_core > 0, compute the frozen core energy and transform one-electron integrals

    frozen_core_energy = 0

    if frozen_core > 0:
        for i in range(frozen_core):
            frozen_core_energy += 2 * mo_oeis[i, i]

        # Note that the two-electron integrals out of Psi4 are in the Mulliken notation
        for i in range(frozen_core):
            for j in range(frozen_core):
                frozen_core_energy += 2 * mo_teis[i, i, j, j] - mo_teis[i, j, j, i]

        # Incorporate in the one-electron integrals the two-electron integrals involving both frozen and non-frozen orbitals.
        # This also ensures that the correct orbital energies will be obtained.

        for p in range(frozen_core, nmo - frozen_virtual):
            for q in range(frozen_core, nmo - frozen_virtual):
                for i in range(frozen_core):
                    mo_oeis[p, q] += 2 * mo_teis[p, q, i, i] - mo_teis[p, i, i, q]

    # Make hf_reference
    hf_reference = [1] * (nel - 2 * frozen_core) + [0] * (2 * (nmo - frozen_virtual) - nel)

    # Build second quantized Hamiltonian
    Hsq = qforte.SQOperator()
    Hsq.add(p4_Enuc_ref + frozen_core_energy, [], [])
    for i in range(frozen_core, nmo - frozen_virtual):
        ia = (i - frozen_core)*2
        ib = (i - frozen_core)*2 + 1
        for j in range(frozen_core, nmo - frozen_virtual):
            ja = (j - frozen_core)*2
            jb = (j - frozen_core)*2 + 1

            Hsq.add(mo_oeis[i,j], [ia], [ja])
            Hsq.add(mo_oeis[i,j], [ib], [jb])

            for k in range(frozen_core, nmo - frozen_virtual):
                ka = (k - frozen_core)*2
                kb = (k - frozen_core)*2 + 1
                for l in range(frozen_core, nmo - frozen_virtual):
                    la = (l - frozen_core)*2
                    lb = (l - frozen_core)*2 + 1

                    if(ia!=jb and kb != la):
                        Hsq.add( mo_teis[i,l,k,j]/2, [ia, jb], [kb, la] ) # abba
                    if(ib!=ja and ka!=lb):
                        Hsq.add( mo_teis[i,l,k,j]/2, [ib, ja], [ka, lb] ) # baab

                    if(ia!=ja and ka!=la):
                        Hsq.add( mo_teis[i,l,k,j]/2, [ia, ja], [ka, la] ) # aaaa
                    if(ib!=jb and kb!=lb):
                        Hsq.add( mo_teis[i,l,k,j]/2, [ib, jb], [kb, lb] ) # bbbb

    # Set attributes
    qforte_mol.nuclear_repulsion_energy = p4_Enuc_ref
    qforte_mol.hf_energy = p4_Escf
    qforte_mol.hf_reference = hf_reference
    qforte_mol.sq_hamiltonian = Hsq
    qforte_mol.hamiltonian = Hsq.jw_transform()
    qforte_mol.point_group = [point_group, irreps]
    qforte_mol.orb_irreps = orb_irreps
    qforte_mol.orb_irreps_to_int = orb_irreps_to_int
    qforte_mol.hf_orbital_energies = hf_orbital_energies
    qforte_mol.frozen_core = frozen_core
    qforte_mol.frozen_virtual = frozen_virtual
    qforte_mol.frozen_core_energy = frozen_core_energy

    # Order Psi4 to delete its temporary files.
    psi4.core.clean()

    return qforte_mol


def create_external_mol(**kwargs):
    """Builds a qforte Molecule object from an external json file containing
    the one and two electron integrals and numbers of alpha/beta electrons.

    Returns
    -------
    Molecule
        The qforte Molecule object which holds the molecular information.
    """

    qforte_mol = Molecule(multiplicity = kwargs['multiplicity'],
                                charge = kwargs['charge'],
                                filename = kwargs['filename'])

    # open json file
    with open(kwargs["filename"]) as f:
        external_data = json.load(f)

    # build sq hamiltonian
    qforte_sq_hamiltonian = qforte.SQOperator()
    qforte_sq_hamiltonian.add(external_data['scalar_energy']['data'], [], [])

    for p, q, h_pq in external_data['oei']['data']:
        qforte_sq_hamiltonian.add(h_pq, [p], [q])

    for p, q, r, s, h_pqrs in external_data['tei']['data']:
        qforte_sq_hamiltonian.add(h_pqrs/4.0, [p,q], [s,r]) # only works in C1 symmetry

    hf_reference = [0 for i in range(external_data['nso']['data'])]
    for n in range(external_data['na']['data'] + external_data['nb']['data']):
        hf_reference[n] = 1

    qforte_mol.point_group = ['C1', 'A']
    qforte_mol.orb_irreps = ['A'] * external_data['nso']['data']
    qforte_mol.orb_irreps_to_int = [0] * external_data['nso']['data']

    qforte_mol.hf_reference = hf_reference

    qforte_mol.sq_hamiltonian = qforte_sq_hamiltonian

    qforte_mol.hamiltonian = qforte_sq_hamiltonian.jw_transform()

    return qforte_mol
