from qforte.adapters import molecule_adapters as MA
from qforte.adapters import model_adapters as mod


def system_factory(system_type="molecule", build_type="psi4", **kwargs):
    """Builds an empty system object of type ('molecule' or 'model') using
    adapters specified by build_type.

     Arguments
     ---------
     system_type : {"molecule"}
         Gives the type of system object to return.

     build_type : {"external", "psi4"}
         Specifies the adapter used to build the system.

     Returns
     -------
     my_sys_skeleton : MolAdapter
         A molecular/hubbard/jellium... adapter object which can be used to
         populate the system info.

    """

    molecule_adapters = {"external": MA.create_external_mol, "psi4": MA.create_psi_mol}

    model_adapters = {"TFIM": mod.create_TFIM}

    if system_type == "molecule":
        kwargs.setdefault("basis", "sto-3g")
        kwargs.setdefault("multiplicity", 1)
        kwargs.setdefault("charge", 0)
        kwargs.setdefault("symmetry", "C1")
        kwargs.setdefault("description", "")
        kwargs.setdefault("filename", "output")
        kwargs.setdefault("hdf5_dir", None)
        kwargs.setdefault("json_dump", False)
        kwargs.setdefault("dipole", False)
        try:
            adapter = molecule_adapters[build_type]
        except:
            raise TypeError(
                f"build type {build_type} not supported, supported types are: "
                + ", ".join(molecule_adapters.keys())
            )
    elif system_type == "model":
        try:
            adapter = model_adapters[build_type]
        except:
            raise TypeError(
                f"build type {build_type} not supported, supported types are: "
                + ", ".join(model_adapters.keys())
            )

    else:
        raise TypeError(
            "system type not supported, supported types are 'molecule' and 'model'."
        )

    return adapter(**kwargs)
