"""
A class for the system information, either a molecule or a lattice system such as
Hubbard model.
"""

# TODO: Documentation needs to be fixed, attributes should be listed below
#       as opposed to arguments for __init__() (Nick).

class Molecule(object):
    """Class for storing moleucular information. Should be instatiated using using
    a MolAdapter and populated by calling MolAdapter.run(**kwargs).


    Atributes
    ---------
    _mol_geometry : list of tuples
        Gives coordinates of each atom in Angstroms. Example format is
        geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 1.50))].

    _basis : string
        Gives the basis set to be used. Default is 'sto-3g'.

    _multiplicity : int
        Gives the targeted spin multiplicity of the molecular system.

    _charge : int
        Gives the targeted net charge of the molecular system (controls number of
        electrons to be considered).

    _description : optional, string
        Recommended to use to distinguish various runs
        (for example with different bond lengths or geometric configurations),
        if populated using a OpenFermionMolAdapter.

    _filename : optional, string
        Specifies the name of the .hdf5 file molecular data from psi4/pyscf
        calculation will be stored in, if populated using a
        OpenFermionMolAdapter.

    _hdf5_dir : optional, string
        Specifies the directory in which to store the .hdf5 file molecular
        data from psi4/pyscf calculation will be stored in.
        Default is "<openfermion_src_dir>/data", if populated using a
        OpenFermionMolAdapter.

    """

    def __init__(self, mol_geometry=None, basis='sto-3g', multiplicity=1, charge=0,
                 description="", filename="", hdf5_dir=None   ):
        """Initialize a qforte molecule object.

        Arguments
        ---------
        mol_geometry : tuple of tuples
            Gives the coordinates of each atom in the moleucle.
            An example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
            Distances in angstrom.

        basis : string
            Gives the basis set. Default is 'sto-3g'.

        charge : int
            Gives the total molecular charge. Defaults to 0.

        multiplicity : int
            Gives the spin multiplicity.

        description : optional, string
            Gives a description of the molecule.

        filename : optional, string
            Gives name of file to use if generating with OpenFermion-Psi4
            or OpenFermion-pyscf.

        hdf5_dir : optional, string
            Optional data directory to change from default
            data directory specified in config file if generating with
            OpenFermion-Psi4 or OpenFermion-pyscf.
        """

        self.geometry = mol_geometry
        self.basis = basis
        self.multiplicity = multiplicity
        self.charge = charge
        self.description = description
        self.filename = filename
        self.hdf5_dir = hdf5_dir

    def set_hamiltonian(self, hamiltonian_operator):
        self._hamiltonian = hamiltonian_operator

    def set_sq_hamiltonian(self, sq_hamiltonian_operator):
        self._sq_hamiltonian = sq_hamiltonian_operator

    def set_sq_of_ham(self, sq_of_ham):
        self._sq_of_ham = sq_of_ham

    def set_nmo(self, nmo):
        self._nmo = nmo

    def set_nel(self, nel):
        self._nel = nel

    def set_hf_reference(self, hf_reference):
        self._hf_reference = hf_reference

    def set_mo_oei(self, oei):
        self._mo_oei = oei

    def set_mo_tei(self, tei):
        self._mo_tei = tei

    def set_ccsd_amps(self, ccsd_singles, ccsd_doubles):
        self._ccsd_singles = ccsd_singles
        self._ccsd_doubles = ccsd_doubles

    def set_nuclear_repulsion_energy(self, nuc_rep_energy):
        self._nuc_rep_energy = nuc_rep_energy

    def set_hf_energy(self, hf_energy):
        self._hf_energy = hf_energy

    def set_mp2_energy(self, mp2_energy):
        self._mp2_energy = mp2_energy

    def set_cisd_energy(self, cisd_energy):
        self._cisd_energy = cisd_energy

    def set_ccsd_energy(self, ccsd_energy):
        self._ccsd_energy = ccsd_energy

    def set_fci_energy(self, fci_energy):
        self._fci_energy = fci_energy


    def get_ccsd_amps(self):
        return self._ccsd_singles, self._ccsd_doubles

    def get_hamiltonian(self):
        return self._hamiltonian

    def get_sq_hamiltonian(self):
        return self._sq_hamiltonian

    def get_sq_of_ham(self):
        return self._sq_of_ham

    def get_hf_reference(self):
        return self._hf_reference

    def get_hf_energy(self):
        return self._hf_energy

    def get_mp2_energy(self):
        return self._mp2_energy

    def get_cisd_energy(self):
        return self._cisd_energy

    def get_ccsd_energy(self):
        return self._ccsd_energy

    def get_fci_energy(self):
        return self._fci_energy
