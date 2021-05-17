"""
A class for the building molecular object adapters. Adapters for various approaches to build
the molecular info and properties (hamiltonian, rdms, etc...).
"""
import operator
import numpy as np
from abc import ABC, abstractmethod

import qforte

from qforte.helper.operator_helper import build_from_openfermion, build_sqop_from_openfermion
from qforte.system.molecular_info import Molecule
from qforte.utils import transforms as tf

from openfermion.ops import FermionOperator, QubitOperator

try:
    from openfermion.chem import MolecularData
except:
    from openfermion.hamiltonians import MolecularData

from openfermion.transforms import get_fermion_operator, jordan_wigner

try:
    from openfermion.transforms.opconversions import normal_ordered
    from openfermion.transforms.repconversions import freeze_orbitals
    from openfermion.utils import hermitian_conjugated
except:
    from openfermion.utils import hermitian_conjugated, normal_ordered, freeze_orbitals

from openfermionpsi4 import run_psi4

import json

try:
    import psi4
    use_psi4 = True
except:
    use_psi4 = False


def create_openfermion_mol(**kwargs):
    """Builds a Molecule object using openfermion as a backend.
    By default, it runs a scf calcuation and stores the qubit hamiltonian
    (hamiltonian in poly word representation).

    Variables
    ---------
    mol_geometry : list of tuples
        Gives coordinates of each atom in Angstroms. Example format is
        geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 1.50))]. It serves
        as an argument for the MolecularData class in openfermion.

    basis : string
        Gives the basis set to be used. Default is 'sto-3g'. It serves
        as an argument for the MolecularData class in openfermion.

    multiplicity : int
        Gives the targeted spin multiplicity of the molecular system. It serves
        as an argument for the MolecularData class in openfermion.

    charge : int
        Gives the targeted net charge of the molecular system (controls number of
        electrons to be considered). It serves as an argument for the
        MolecularData class in openfermion.

    description : optional, string
        Recomeded to use to distingush various runs
        (for example with differnet bond lengths or geometric configurations).

    filename : optional, string
        Specifies the name of the .hdf5 file molecular data from psi4/pyscf
        calculation will be stored in.

    hdf5_dir : optional, string
        Specifies the directory in which to store the .hdf5 file molecular
        data from psi4/pyscf calculation will be stored in.
        Default is "<openfermion_src_dir>/data".

    Returns
    -------
    Molecule
        The qforte Molecule object which holds the molecular information.
    """

    qforte_mol = Molecule(mol_geometry = kwargs['mol_geometry'],
                               basis = kwargs['basis'],
                               multiplicity = kwargs['multiplicity'],
                               charge = kwargs['charge'],
                               description = kwargs['description'],
                               filename = kwargs['filename'],
                               hdf5_dir = kwargs['hdf5_dir'])

    kwargs.setdefault('order_sq_ham', False)
    kwargs.setdefault('order_jw_ham', False)
    kwargs.setdefault('run_scf', 1)
    kwargs.setdefault('run_mp2', 0)
    kwargs.setdefault('run_ccsd', 0)
    kwargs.setdefault('run_cisd', 0)
    kwargs.setdefault('run_cisd', 0)
    kwargs.setdefault('run_fci', 1)
    kwargs.setdefault('store_uccsd_amps', False)
    kwargs.setdefault('frozen_indices', None)
    kwargs.setdefault('virtual_indices', None)

    skeleton_mol = MolecularData(geometry = kwargs["mol_geometry"],
                                 basis = kwargs["basis"],
                                 multiplicity = kwargs["multiplicity"],
                                 charge = kwargs["charge"],
                                 description = kwargs["description"],
                                 filename = kwargs["filename"],
                                 data_directory = kwargs["hdf5_dir"])

    openfermion_mol = run_psi4(skeleton_mol, run_scf=kwargs['run_scf'],
                                             run_mp2=kwargs['run_mp2'],
                                             run_ccsd=kwargs['run_ccsd'],
                                             run_cisd=kwargs['run_cisd'],
                                             run_fci=kwargs['run_fci'])

    openfermion_mol.load()

    # Set qforte hamiltonian from openfermion
    molecular_hamiltonian = openfermion_mol.get_molecular_hamiltonian()

    if(kwargs['frozen_indices'] is not None or kwargs['virtual_indices'] is not None):
        if kwargs['frozen_indices'] is None:
            # We want to freeze virtuals but not core. Openfermion requires frozen_indices to be non-empty.
            # As of 5/5/21, this is because freeze_orbitals assumes you can call "in" on the frozen_indices.
            kwargs['frozen_indices'] = []
        if any(x >= molecular_hamiltonian.n_qubits for x in kwargs['frozen_indices'] + kwargs['virtual_indices']):
            raise ValueError(f"The orbitals to freeze are inconsistent with the fact that we only have {molecular_hamiltonian.n_qubits} qubits.")

        fermion_hamiltonian = normal_ordered(
            freeze_orbitals(get_fermion_operator(molecular_hamiltonian),
                            kwargs['frozen_indices'],
                            unoccupied=kwargs['virtual_indices']))
    else:
        fermion_hamiltonian = normal_ordered(get_fermion_operator(molecular_hamiltonian))

    if(kwargs['order_sq_ham'] or kwargs['order_jw_ham']):

        if(kwargs['order_sq_ham'] and kwargs['order_jw_ham']):
            raise ValueError("Can't use more than one hamiltonian ordering option!")

        if(kwargs['order_sq_ham']):
            # Optionally sort the hamiltonian terms
            print('using |largest|->|smallest| sq hamiltonian ordering!')
            sorted_terms = sorted(fermion_hamiltonian.terms.items(), key=lambda kv: np.abs(kv[1]), reverse=True)

            # Try converting with qforte jw functions
            sorted_sq_excitations = tf.fermop_to_sq_excitation(sorted_terms)
            sorted_organizer = tf.get_ucc_jw_organizer(sorted_sq_excitations, already_anti_herm=True)
            qforte_hamiltonian = tf.organizer_to_circuit(sorted_organizer)

        if(kwargs['order_jw_ham']):
            print('using |largest|->|smallest| jw hamiltonian ordering!')
            unsorted_terms = [(k, v) for k, v in fermion_hamiltonian.terms.items()]

            # Try converting with qforte jw functions
            unsorted_sq_excitations = tf.fermop_to_sq_excitation(unsorted_terms)
            # print('\nunsorted_sq_excitations:\n', unsorted_sq_excitations)

            unsorted_organizer = tf.get_ucc_jw_organizer(unsorted_sq_excitations, already_anti_herm=True)
            # print('\nunsorted_organizer:\n', unsorted_organizer)

            # Sort organizer
            sorted_organizer = sorted(unsorted_organizer, key = lambda x: np.abs(x[0]), reverse=True)


            qforte_hamiltonian = tf.organizer_to_circuit(sorted_organizer)

    else:
        print('Using standard openfermion hamiltonian ordering!')
        qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
        qforte_hamiltonian = build_from_openfermion(qubit_hamiltonian)

    qforte_mol.set_hamiltonian(qforte_hamiltonian)

    qforte_mol.set_sq_hamiltonian( build_sqop_from_openfermion(fermion_hamiltonian) )

    qforte_mol.set_sq_of_ham(fermion_hamiltonian)

    # Set qforte energies from openfermion
    if(kwargs['run_scf']):
        qforte_mol.set_hf_energy(openfermion_mol.hf_energy)

    if(kwargs['run_mp2']):
        qforte_mol.set_mp2_energy(openfermion_mol.mp2_energy)

    if(kwargs['run_cisd']):
        qforte_mol.set_cisd_energy(openfermion_mol.cisd_energy)

    if(kwargs['run_ccsd']):
        qforte_mol.set_ccsd_energy(openfermion_mol.ccsd_energy)

        # Store uccsd circuit with initial guess from ccsd amplitudes
        if(kwargs['store_uccsd_amps']==True):
            qforte_mol.set_ccsd_amps(openfermion_mol.ccsd_single_amps,
                          openfermion_mol.ccsd_double_amps)

    if(kwargs['run_fci']):
        qforte_mol.set_fci_energy(openfermion_mol.fci_energy)

    return qforte_mol


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

    self._qforte_mol = Molecule(mol_geometry = mol_geometry,
                               basis = basis,
                               multiplicity = multiplicity,
                               charge = charge)

    if not use_psi4:
        raise ImportError("Psi4 was not imported correctely.")

    # run_scf is not read, because we always run SCF to get a wavefunction object.
    kwargs.setdefault('run_mp2', 0)
    kwargs.setdefault('run_ccsd', 0)
    kwargs.setdefault('run_cisd', 0)
    kwargs.setdefault('run_fci', 1)

    # Setup psi4 calculation(s)
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('output.dat', False)

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

    psi4.set_options({'basis': self._basis,
              'scf_type': 'pk',
              'reference' : scf_ref_type,
              'e_convergence': 1e-8,
              'd_convergence': 1e-8,
              'ci_maxiter': 100})

    # run psi4 caclulation
    p4_Escf, p4_wfn = psi4.energy('SCF', return_wfn=True)

    if(kwargs['run_mp2']):
        qforte_mol.set_mp2_energy(psi4.energy('MP2'))

    if(kwargs['run_ccsd']):
        qforte_mol.set_ccsd_energy(psi4.energy('CCSD'))

    if(kwargs['run_cisd']):
        qforte_mol.set_cisd_energy(psi4.energy('CISD'))

    if(kwargs['run_fci']):
        qforte_mol.set_fci_energy(psi4.energy('FCI'))

    # Get integrals using MintsHelper.
    mints = psi4.core.MintsHelper(p4_wfn.basisset())

    C = p4_wfn.Ca()
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

    # Make hf_reference
    hf_reference = [1] * nel + [0] * (2 * nmo - nel)

    # Build second quantized Hamiltonian
    nmo = np.shape(mo_oeis)[0]
    Hsq = qforte.SQOperator()
    Hsq.add_term(p4_Enuc_ref, [])
    for i in range(nmo):
        ia = i*2
        ib = i*2 + 1
        for j in range(nmo):
            ja = j*2
            jb = j*2 + 1

            Hsq.add_term(mo_oeis[i,j], [ia, ja])
            Hsq.add_term(mo_oeis[i,j], [ib, jb])

            for k in range(nmo):
                ka = k*2
                kb = k*2 + 1
                for l in range(nmo):
                    la = l*2
                    lb = l*2 + 1

                    if(ia!=jb and kb != la):
                        Hsq.add_term( mo_teis[i,l,k,j]/2, [ia, jb, kb, la] ) # abba
                    if(ib!=ja and ka!=lb):
                        Hsq.add_term( mo_teis[i,l,k,j]/2, [ib, ja, ka, lb] ) # baab

                    if(ia!=ja and ka!=la):
                        Hsq.add_term( mo_teis[i,l,k,j]/2, [ia, ja, ka, la] ) # aaaa
                    if(ib!=jb and kb!=lb):
                        Hsq.add_term( mo_teis[i,l,k,j]/2, [ib, jb, kb, lb] ) # bbbb

    # Set attributes
    qforte_mol.set_nuclear_repulsion_energy(p4_Enuc_ref)
    qforte_mol.set_hf_energy(p4_Escf)
    qforte_mol.set_hf_reference(hf_reference)
    qforte_mol.set_sq_hamiltonian(Hsq)
    qforte_mol.set_hamiltonian(Hsq.jw_transform())

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
    qforte_sq_hamiltonian.add_term(external_data['scalar_energy']['data'], [])

    for p, q, h_pq in external_data['oei']['data']:
        qforte_sq_hamiltonian.add_term(h_pq, [p,q])

    for p, q, r, s, h_pqrs in external_data['tei']['data']:
        qforte_sq_hamiltonian.add_term(h_pqrs/4.0, [p,q,s,r]) # only works in C1 symmetry

    hf_reference = [0 for i in range(external_data['nso']['data'])]
    for n in range(external_data['na']['data'] + external_data['nb']['data']):
        hf_reference[n] = 1

    qforte_mol.set_hf_reference(hf_reference)

    qforte_mol.set_sq_hamiltonian(qforte_sq_hamiltonian)

    qforte_mol.set_hamiltonian(qforte_sq_hamiltonian.jw_transform())

    return qforte_mol
