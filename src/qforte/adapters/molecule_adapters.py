"""
A class for the building molecular object adapters. Adapters for various approaches to build
the molecular info and properties (hamiltonian, rdms, etc...).
"""

from abc import ABC, abstractmethod

from qforte.helper.operator_helper import build_from_openfermion
from qforte.system.molecular_info import Molecule

from openfermion.ops import QubitOperator
from openfermion.hamiltonians import MolecularData
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermion.utils import hermitian_conjugated, normal_ordered

from openfermionpsi4 import run_psi4

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

        kwargs.setdefault('run_scf', 1)
        kwargs.setdefault('run_mp2', 0)
        kwargs.setdefault('run_ccsd', 0)
        kwargs.setdefault('run_cisd', 0)
        kwargs.setdefault('run_cisd', 0)
        kwargs.setdefault('run_fci', 0)
        kwargs.setdefault('store_uccsd_amps', False)
        kwargs.setdefault('buld_uccsd_circ_from_ccsd', False)

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
        fermion_hamiltonian = normal_ordered(get_fermion_operator(molecular_hamiltonian))
        qubit_hamiltonian= jordan_wigner(fermion_hamiltonian)

        qforte_hamiltionan = build_from_openfermion(qubit_hamiltonian)
        self._qforte_mol.set_hamiltonian(qforte_hamiltionan)

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
