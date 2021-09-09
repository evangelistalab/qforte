"""
Classes for system information, e.g., molecule or Hubbard model.
"""

# TODO: Documentation needs to be fixed, attributes should be listed below
#       as opposed to arguments for __init__() (Nick).

class System(object):
    """Class for a generic quantum many-body system."""

    @property
    def fci_energy(self):
        return self._fci_energy

    @fci_energy.setter
    def fci_energy(self, fci_energy):
        self._fci_energy = fci_energy

    @property
    def hamiltonian(self):
        return self._hamiltonian

    @hamiltonian.setter
    def hamiltonian(self, hamiltonian_operator):
        self._hamiltonian = hamiltonian_operator

    @property
    def sq_hamiltonian(self):
        return self._sq_hamiltonian

    @sq_hamiltonian.setter
    def sq_hamiltonian(self, sq_hamiltonian_operator):
        self._sq_hamiltonian = sq_hamiltonian_operator

    @property
    def hf_reference(self):
        return self._hf_reference

    @hf_reference.setter
    def hf_reference(self, hf_reference):
        self._hf_reference = hf_reference

class Molecule(System):
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

    @property
    def ccsd_amps(self):
        return self._ccsd_singles, self._ccsd_doubles

    @ccsd_amps.setter
    def ccsd_amps(self, ccsd_amps):
        ccsd_singles, ccsd_doubles = ccsd_amps
        self._ccsd_singles = ccsd_singles
        self._ccsd_doubles = ccsd_doubles

    @property
    def hf_energy(self):
        return self._hf_energy

    @hf_energy.setter
    def hf_energy(self, hf_energy):
        self._hf_energy = hf_energy

    @property
    def mp2_energy(self):
        return self._mp2_energy

    @mp2_energy.setter
    def mp2_energy(self, mp2_energy):
        self._mp2_energy = mp2_energy

    @property
    def cisd_energy(self):
        return self._cisd_energy

    @cisd_energy.setter
    def cisd_energy(self, cisd_energy):
        self._cisd_energy = cisd_energy

    @property
    def ccsd_energy(self):
        return self._ccsd_energy

    @ccsd_energy.setter
    def ccsd_energy(self, ccsd_energy):
        self._ccsd_energy = ccsd_energy

    @property
    def point_group(self):
        return self._point_group

    @point_group.setter
    def point_group(self, point_group):
        self._point_group = point_group

    @property
    def orb_irreps(self):
        return self._orb_irreps

    @orb_irreps.setter
    def orb_irreps(self, orb_irreps):
        self._orb_irreps = orb_irreps

    @property
    def orb_irreps_to_int(self):
        return self._orb_irreps_to_int

    @orb_irreps_to_int.setter
    def orb_irreps_to_int(self, orb_irreps_to_int):
        self._orb_irreps_to_int = orb_irreps_to_int
