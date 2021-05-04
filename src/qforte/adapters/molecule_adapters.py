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

class MolAdapter(ABC):
    """Abstract class for the aquiring system information from external electronic
    structure calculations. The run() method calculates the desired properties and
    data for the molecular system. Infromation is then stored in the
    self._qforte_mol attribuite.
    """

    @abstractmethod
    def run(self, **kwargs):
        pass

    @abstractmethod
    def get_molecule(self):
        pass


class OpenFermionMolAdapter(MolAdapter):
    """Class which builds a Molecule object using openfermion as a backend.
    By default, it runs a scf calcuation and stores the qubit hamiltonian
    (hamiltonian in
    poly word representation).

    Atributes
    ---------
    _mol_geometry : list of tuples
        Gives coordinates of each atom in Angstroms. Example format is
        geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 1.50))]. It serves
        as an argument for the MolecularData class in openfermion.

    _basis : string
        Gives the basis set to be used. Default is 'sto-3g'. It serves
        as an argument for the MolecularData class in openfermion.

    _multiplicity : int
        Gives the targeted spin multiplicity of the molecular system. It serves
        as an argument for the MolecularData class in openfermion.

    _charge : int
        Gives the targeted net charge of the molecular system (controls number of
        electrons to be considered). It serves as an argument for the
        MolecularData class in openfermion.

    _description : optional, string
        Recomeded to use to distingush various runs
        (for example with differnet bond lengths or geometric configurations).

    _filename : optional, string
        Specifies the name of the .hdf5 file molecular data from psi4/pyscf
        calculation will be stored in.

    _hdf5_dir : optional, string
        Specifies the directory in which to store the .hdf5 file molecular
        data from psi4/pyscf calculation will be stored in.
        Default is "<openfermion_src_dir>/data".


    _qforte_mol : Molecule
        The qforte Molecule object which holds the molecular information.

    """

    def __init__(self, **kwargs):
        self._mol_geometry = kwargs['mol_geometry']
        self._basis = kwargs['basis']
        self._multiplicity = kwargs['multiplicity']
        self._charge = kwargs['charge']
        self._description = kwargs['description']
        self._filename = kwargs['filename']
        self._hdf5_dir = kwargs['hdf5_dir']

        self._qforte_mol = Molecule(mol_geometry = kwargs['mol_geometry'],
                                   basis = kwargs['basis'],
                                   multiplicity = kwargs['multiplicity'],
                                   charge = kwargs['charge'],
                                   description = kwargs['description'],
                                   filename = kwargs['filename'],
                                   hdf5_dir = kwargs['hdf5_dir'])

    def run(self, **kwargs):

        kwargs.setdefault('order_sq_ham', False)
        kwargs.setdefault('order_jw_ham', False)
        kwargs.setdefault('run_scf', 1)
        kwargs.setdefault('run_mp2', 0)
        kwargs.setdefault('run_ccsd', 0)
        kwargs.setdefault('run_cisd', 0)
        kwargs.setdefault('run_cisd', 0)
        kwargs.setdefault('run_fci', 1)
        kwargs.setdefault('store_uccsd_amps', False)
        kwargs.setdefault('buld_uccsd_circ_from_ccsd', False)
        kwargs.setdefault('frozen_indices', None)
        kwargs.setdefault('virtual_indices', None)

        skeleton_mol = MolecularData(geometry = self._mol_geometry,
                                     basis = self._basis,
                                     multiplicity = self._multiplicity,
                                     charge = self._charge,
                                     description = self._description,
                                     filename = self._filename,
                                     data_directory = self._hdf5_dir)

        openfermion_mol = run_psi4(skeleton_mol, run_scf=kwargs['run_scf'],
                                                 run_mp2=kwargs['run_mp2'],
                                                 run_ccsd=kwargs['run_ccsd'],
                                                 run_cisd=kwargs['run_cisd'],
                                                 run_fci=kwargs['run_fci'])

        openfermion_mol.load()

        # Set qforte hamiltonian from openfermion
        molecular_hamiltonian = openfermion_mol.get_molecular_hamiltonian()

        if(kwargs['frozen_indices'] is not None):
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
                # print('\n\nsorted terms\n\n', sorted_terms)

                # Try converting with qforte jw functions
                sorted_sq_excitations = tf.fermop_to_sq_excitation(sorted_terms)
                # print('\nsorted_sq_excitations:\n', sorted_sq_excitations)
                sorted_organizer = tf.get_ucc_jw_organizer(sorted_sq_excitations, already_anti_herm=True)
                # print('\nsorted_organizer:\n', sorted_organizer)
                qforte_hamiltonian = tf.organizer_to_circuit(sorted_organizer)
                # print('\nqforte_hamiltionan:\n', '  len: ', len(qforte_hamiltionan.terms()))
                # for term in qforte_hamiltionan.terms():
                #     print(term[0])
                #     print(term[1].str())

            if(kwargs['order_jw_ham']):
                print('using |largest|->|smallest| jw hamiltonain ordering!')
                unsorted_terms = [(k, v) for k, v in fermion_hamiltonian.terms.items()]

                # Try converting with qforte jw functions
                unsorted_sq_excitations = tf.fermop_to_sq_excitation(unsorted_terms)
                # print('\nunsorted_sq_excitations:\n', unsorted_sq_excitations)

                unsorted_organizer = tf.get_ucc_jw_organizer(unsorted_sq_excitations, already_anti_herm=True)
                # print('\nunsorted_organizer:\n', unsorted_organizer)

                # Sort organizer
                sorted_organizer = sorted(unsorted_organizer, key = lambda x: np.abs(x[0]), reverse=True)


                qforte_hamiltonian = tf.organizer_to_circuit(sorted_organizer)
                # print('\nqforte_hamiltionan:\n', '  len: ', len(qforte_hamiltionan.terms()))

                # for term in qforte_hamiltionan.terms():
                #     print(term[0])
                #     print(term[1].str())


        else:
            print('Using standard openfermion hamiltonian ordering!')
            qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
            qforte_hamiltonian = build_from_openfermion(qubit_hamiltonian)

            # print('\nqforte_hamiltionan:\n', '  len: ', len(qforte_hamiltionan.terms()))
            # for term in qforte_hamiltionan.terms():
            #     print(term[0])
            #     print(term[1].str())

        self._qforte_mol.set_hamiltonian(qforte_hamiltonian)

        self._qforte_mol.set_sq_hamiltonian( build_sqop_from_openfermion(fermion_hamiltonian) )

        ##
        self._qforte_mol.set_sq_of_ham(fermion_hamiltonian)
        ##

        # Set qforte energies from openfermion
        if(kwargs['run_scf']==1):
            self._qforte_mol.set_hf_energy(openfermion_mol.hf_energy)

        if(kwargs['run_mp2']==1):
            self._qforte_mol.set_mp2_energy(openfermion_mol.mp2_energy)

        if(kwargs['run_cisd']==1):
            self._qforte_mol.set_cisd_energy(openfermion_mol.cisd_energy)

        if(kwargs['run_ccsd']==1):
            self._qforte_mol.set_ccsd_energy(openfermion_mol.ccsd_energy)

            # Store uccsd circuit with initial guess from ccsd amplitudes
            if(kwargs['store_uccsd_amps']==True):
                self._qforte_mol.set_ccsd_amps(openfermion_mol.ccsd_single_amps,
                              openfermion_mol.ccsd_double_amps)

        if(kwargs['run_fci']==1):
            self._qforte_mol.set_fci_energy(openfermion_mol.fci_energy)



    def get_molecule(self):
        return self._qforte_mol

class Psi4MolAdapter(MolAdapter):
    """Builds a qforte Molecule object directly from a psi4 calculation.
    """

    def __init__(self, **kwargs):

        kwargs.setdefault('symmetry', 'c1')
        kwargs.setdefault('charge', 0)
        kwargs.setdefault('multiplicity', 1)


        self._mol_geometry = kwargs['mol_geometry']
        self._basis = kwargs['basis']
        self._multiplicity = kwargs['multiplicity']
        self._charge = kwargs['charge']
        self._symmetry = kwargs['symmetry']


        self._qforte_mol = Molecule(mol_geometry = kwargs['mol_geometry'],
                                   basis = kwargs['basis'],
                                   multiplicity = kwargs['multiplicity'],
                                   charge = kwargs['charge'])


    def run(self, **kwargs):

        if not use_psi4:
            raise ImportError("Psi4 was not imported correctely.")

        kwargs.setdefault('run_scf', 1)
        kwargs.setdefault('run_mp2', 0)
        kwargs.setdefault('run_ccsd', 0)
        kwargs.setdefault('run_cisd', 0)
        kwargs.setdefault('run_fci', 1)
        kwargs.setdefault('e_and_d_converge', 1e-8)

        self._e_and_d_converge = kwargs['e_and_d_converge']

        # Setup psi4 calcualtion(s)
        psi4.set_memory('2 GB')
        psi4.core.set_output_file('psi4_output.dat', False)

        p4_geom_str =  f"{int(self._charge)}  {int(self._multiplicity)}"
        for geom_line in self._mol_geometry:
            p4_geom_str += f"\n{geom_line[0]}  {geom_line[1][0]}  {geom_line[1][1]}  {geom_line[1][2]}"
        p4_geom_str += f"\nsymmetry {self._symmetry}"
        p4_geom_str += f"\nunits angstrom"

        print(' ==> Psi4 geometry <==')
        print('-------------------------')
        print(p4_geom_str)
        print('\n')
        print(f' Psi4 SCF Econv: {self._e_and_d_converge}')
        print(f' Psi4 SCF Dconv: {self._e_and_d_converge}')

        p4_mol = psi4.geometry(p4_geom_str)

        if self._multiplicity == 1:
            scf_ref_type = 'rhf'
        else:
            scf_ref_type = 'rohf'

        psi4.set_options({'basis': self._basis,
                  'scf_type': 'pk',
                  'reference' : scf_ref_type,
                  'e_convergence': self._e_and_d_converge,
                  'd_convergence': self._e_and_d_converge,
                  'ci_maxiter': 100})

        # run psi4 caclulation
        if(kwargs['run_scf']):
            p4_Escf, p4_wfn = psi4.energy('SCF', return_wfn=True)

        if(kwargs['run_mp2']):
            p4_Emp2 = psi4.energy('MP2')

        if(kwargs['run_ccsd']):
            p4_Eccsd = psi4.energy('CCSD')

        if(kwargs['run_ccsd']):
            p4_Ecisd = psi4.energy('CISD')

        if(kwargs['run_fci']):
            p4_Efci = psi4.energy('FCI')

        # Get integrals using MintsHelper.
        mints = psi4.core.MintsHelper(p4_wfn.basisset())

        C = p4_wfn.Ca()
        scalars = p4_wfn.scalar_variables()

        # print(C.to_array().dot(C.to_array().transpose()))
        print(C.to_array())

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
        hf_reference = []
        for a in range(2*nmo):
            if(a<nel):
                hf_reference.append(1)
            else:
                hf_reference.append(0)

        # Build second quantized Hamiltonain
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
        self._qforte_mol.set_nuclear_repulsion_energy(p4_Enuc_ref)
        self._qforte_mol.set_hf_energy(p4_Escf)
        self._qforte_mol.set_hf_reference(hf_reference)
        self._qforte_mol.set_mo_oei(mo_oeis)
        self._qforte_mol.set_mo_tei(mo_teis)
        self._qforte_mol.set_sq_hamiltonian(Hsq)
        self._qforte_mol.set_hamiltonian(Hsq.jw_transform())

        if(kwargs['run_mp2']):
            self._qforte_mol.set_mp2_energy(p4_Emp2)

        if(kwargs['run_cisd']):
            self._qforte_mol.set_cisd_energy(p4_Ecisd)

        if(kwargs['run_ccsd']):
            self._qforte_mol.set_ccsd_energy(p4_Eccsd)

        if(kwargs['run_fci']):
            self._qforte_mol.set_fci_energy(p4_Efci)

    def get_molecule(self):
        return self._qforte_mol

class ExternalMolAdapter(MolAdapter):
    """Builds a qforte Molecule object from an external json file containing
    the one and two electron integrals and numbers of alpha/beta electrons.
    """

    def __init__(self, **kwargs):
        # self._basis = kwargs['basis']
        self._multiplicity = kwargs['multiplicity']
        self._charge = kwargs['charge']
        self._filename = kwargs['filename']


        self._qforte_mol = Molecule(multiplicity = kwargs['multiplicity'],
                                    charge = kwargs['charge'],
                                    filename = kwargs['filename'])

    def run(self, **kwargs):

        # open json file
        with open(self._filename) as f:
            external_data = json.load(f)

        # build sq hamiltonain
        qforte_sq_hamiltionan = qforte.SQOperator()
        qforte_sq_hamiltionan.add_term(external_data['scalar_energy']['data'], [])

        for p, q, h_pq in external_data['oei']['data']:
            qforte_sq_hamiltionan.add_term(h_pq, [p,q])

        for p, q, r, s, h_pqrs in external_data['tei']['data']:
            # qforte_sq_hamiltionan.add_term(h_pqrs, [p,q,r,s])
            qforte_sq_hamiltionan.add_term(h_pqrs/4.0, [p,q,s,r]) # only works in C1 symmetry

        hf_reference = [0 for i in range(external_data['nso']['data'])]
        for n in range(external_data['na']['data'] + external_data['nb']['data']):
            hf_reference[n] = 1

        self._qforte_mol.set_hf_reference(hf_reference)

        self._qforte_mol.set_sq_hamiltonian(qforte_sq_hamiltionan)

        self._qforte_mol.set_hamiltonian(qforte_sq_hamiltionan.jw_transform())


    def get_molecule(self):
        return self._qforte_mol
