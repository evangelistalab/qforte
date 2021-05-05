from qforte.adapters import molecule_adapters as MA

def system_factory(system_type = 'molecule', build_type = 'openfermion', **kwargs):

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

    adapters = {
        "openfermion": MA.OpenFermionMolAdapter,
        "external": MA.ExternalMolAdapter,
        "psi4": MA.Psi4MolAdapter
    }

    if (system_type=='molecule'):
        try:
            adapter = adapters[build_type]
        except:
            raise TypeError(f"build type {build_type} not supported, supported types are: " + ", ".join(adapters.keys()))

        my_system_skeleton = adapter(**kwargs)

    else:
        raise TypeError("system type not supported, supported type is 'molecule'.")

    return my_system_skeleton
