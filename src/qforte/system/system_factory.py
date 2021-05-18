from qforte.adapters import molecule_adapters as MA

def system_factory(system_type = 'molecule', build_type = 'openfermion', **kwargs):

    """Builds an empty system object of type ('molecule', 'hubbard', 'jellium', etc...) using
       adapters specified by build_type.

        Arguments
        ---------
        system_type : {"molecule"}
            Gives the type of system object to return.

        build_type : {"openfermion", "external", "psi4"}
            Specifies the adapter used to build the system.

        Returns
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

    adapters = {
        "openfermion": MA.create_openfermion_mol,
        "external": MA.create_external_mol,
        "psi4": MA.create_psi_mol
    }

    if (system_type=='molecule'):
        try:
            adapter = adapters[build_type]
        except:
            raise TypeError(f"build type {build_type} not supported, supported types are: " + ", ".join(adapters.keys()))

    else:
        raise TypeError("system type not supported, supported type is 'molecule'.")

    return adapter(**kwargs)
