from qforte.adapters import molecule_adapters as MA

def system_factory(stytem_type = 'molecule', build_type = 'openfermion', **kwargs):

    """Builds an empty system object of type ('molecule', 'hubbard', 'jellium', etc...) using
       adapters specified by build_type.

        Arguments
        ---------
        system_type : string
            Gives the type of system object to return.

        build_type : string
            Gives the method used to construct the system object. For example,
            one could use OpenFermion, or one could just port directly from
            Psi4 or pyscf.

        Retruns
        -------
        my_sys_skeleton : MolAdapter
            A molecular/hubbard/jellium... adapter object which can be used to
            populate the system info.

    """

    kwargs.setdefault('basis', 'sto-3g')
    kwargs.setdefault('multiplicity', 1)
    kwargs.setdefault('charge', 0)
    kwargs.setdefault('description', "")
    kwargs.setdefault('filename', "")
    kwargs.setdefault('hdf5_dir', None)

    if (stytem_type=='molecule'):
        if (build_type=='openfermion'):
            my_system_skeleton = MA.OpenFermionMolAdapter(mol_geometry = kwargs['mol_geometry'],
                                                          basis = kwargs['basis'],
                                                          multiplicity = kwargs['multiplicity'],
                                                          charge = kwargs['charge'],
                                                          description = kwargs['description'],
                                                          filename = kwargs['filename'],
                                                          hdf5_dir = kwargs['hdf5_dir'])

        elif(build_type=='external'):
            my_system_skeleton = MA.ExternalMolAdapter(multiplicity = kwargs['multiplicity'],
                                                       charge = kwargs['charge'],
                                                       filename = kwargs['filename'])

        elif(build_type=='psi4'):
            my_system_skeleton = MA.Psi4MolAdapter(mol_geometry = kwargs['mol_geometry'],
                                                   basis = kwargs['basis'],
                                                   multiplicity = kwargs['multiplicity'],
                                                   charge = kwargs['charge'])

        else:
            raise TypeError("build type not supported, supported type is 'open_fermion'.")

    else:
        raise TypeError("system type not supported, supported type is 'molecule'.")

    return my_system_skeleton
