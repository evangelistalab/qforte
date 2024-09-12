"""
A class for building molecular object adapters. Adapters for various approaches to build
the molecular info and properties (hamiltonian, rdms, etc...).
"""

import numpy as np
import qforte
from qforte.system.molecular_info import Molecule
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

    kwargs.setdefault("symmetry", "c1")
    kwargs.setdefault("charge", 0)
    kwargs.setdefault("multiplicity", 1)

    mol_geometry = kwargs["mol_geometry"]
    basis = kwargs["basis"]
    multiplicity = kwargs["multiplicity"]
    charge = kwargs["charge"]
    json_dump = kwargs["json_dump"]
    dipole = kwargs["dipole"]

    qforte_mol = Molecule(
        mol_geometry=mol_geometry, basis=basis, multiplicity=multiplicity, charge=charge
    )

    if not use_psi4:
        raise ImportError("Psi4 was not imported correctly.")

    # The irreps to be doubly occupied in SCF.
    # e.g. for BeH2 in C2v, we could have:
    # [2, 0, 0, 1] for an A1A1B2 determinant.
    kwargs.setdefault("scf_docc", None)

    # Tuple containing restricted occupied, active, and restricted virtual irrep indices
    # e.g. H4 in D2h symmetry with B2u and B3u orbitals as the active space is
    # ([1,0,...0],[0,...,1,1],[0,1,0,...0])
    kwargs.setdefault("casscf", None)

    # Avoid rotations of the molecule
    kwargs.setdefault("no_reorient", False)
    # Avoid translations of the molecule
    kwargs.setdefault("no_com", False)

    # By default, the number of frozen orbitals is set to zero
    kwargs.setdefault("num_frozen_docc", 0)
    kwargs.setdefault("num_frozen_uocc", 0)

    # run_scf is not read, because we always run SCF to get a wavefunction object.
    kwargs.setdefault("run_mp2", False)
    kwargs.setdefault("run_ccsd", False)
    kwargs.setdefault("run_cisd", False)
    kwargs.setdefault("run_fci", False)

    # Setup psi4 calculation(s)
    psi4.set_memory("2 GB")
    psi4.core.set_output_file(kwargs["filename"] + ".out", False)

    p4_geom_str = f"{int(charge)}  {int(multiplicity)}"
    for geom_line in mol_geometry:
        p4_geom_str += (
            f"\n{geom_line[0]}  {geom_line[1][0]}  {geom_line[1][1]}  {geom_line[1][2]}"
        )
    p4_geom_str += f"\nsymmetry {kwargs['symmetry']}"
    p4_geom_str += f"\nunits angstrom"
    if kwargs["no_reorient"] == True:
        p4_geom_str += f"\nno_reorient"
    if kwargs["no_com"] == True:
        p4_geom_str += f"\nno_com"

    print(" ==> Psi4 geometry <==")
    print("-------------------------")
    print(p4_geom_str)

    p4_mol = psi4.geometry(p4_geom_str)

    scf_ref_type = "rhf" if multiplicity == 1 else "rohf"

    psi4.set_options(
        {
            "basis": basis,
            "scf_type": "pk",
            "reference": scf_ref_type,
            "e_convergence": 1e-8,
            "d_convergence": 1e-8,
            "ci_maxiter": 100,
            "num_frozen_docc": kwargs["num_frozen_docc"],
            "num_frozen_uocc": kwargs["num_frozen_uocc"],
            "mp2_type": "conv",
        }
    )

    if kwargs["scf_docc"] != None:
        psi4.set_options({"docc": kwargs["scf_docc"]})

    if kwargs["num_frozen_docc"] != 0 and kwargs["casscf"] != None:
        print("CASSCF not tested with completely frozen (unmixed) orbitals")
        exit()

    if kwargs["num_frozen_uocc"] != 0 and kwargs["casscf"] != None:
        print("CASSCF not tested with completely frozen (unmixed) orbitals")
        exit()

    # run psi4 caclulation

    if kwargs["casscf"] != None:
        p4_Escf, vanilla_wfn = psi4.energy("SCF", return_wfn=True)
        qforte_mol.fci_energy = psi4.energy("FCI")
        psi4.set_options({"restricted_docc": kwargs["casscf"][0]})
        psi4.set_options({"active": kwargs["casscf"][1]})
        psi4.set_options({"diag_method": "rsp"})
        psi4.set_options({"mcscf_r_convergence": 1e-12})
        psi4.set_options({"mcscf_maxiter": 1000})
        psi4.set_options({"mcscf_diis_start": 50})
        E_casscf, p4_wfn = psi4.energy("casscf", return_wfn=True, ref_wfn=vanilla_wfn)
        p4_Escf = None
        print(f"CASSCF Energy: {E_casscf}")
    else:
        p4_Escf, p4_wfn = psi4.energy("SCF", return_wfn=True)

    # Run additional computations requested by the user
    if kwargs["run_mp2"]:
        qforte_mol.mp2_energy = psi4.energy("MP2")

    if kwargs["run_ccsd"]:
        qforte_mol.ccsd_energy = psi4.energy("CCSD")

    if kwargs["run_cisd"]:
        qforte_mol.cisd_energy = psi4.energy("CISD")

    if kwargs["run_fci"] and kwargs["casscf"] == None:
        if kwargs["num_frozen_uocc"] == 0:
            qforte_mol.fci_energy = psi4.energy("FCI")
        else:
            print(
                "\nWARNING: Skipping FCI computation due to a Psi4 bug related to FCI with frozen virtuals.\n"
            )

    # Get integrals using MintsHelper.
    mints = psi4.core.MintsHelper(p4_wfn.basisset())

    C = p4_wfn.Ca_subset("AO", "ALL")

    scalars = p4_wfn.scalar_variables()

    p4_Enuc_ref = scalars["NUCLEAR REPULSION ENERGY"]

    p4_dip_nuc = p4_mol.nuclear_dipole()

    # Do MO integral transformation
    mo_teis = np.asarray(mints.mo_eri(C, C, C, C))
    mo_oeis = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
    mo_oeis = np.einsum("uj,vi,uv", C, C, mo_oeis)

    nmo = np.shape(mo_oeis)[0]

    nalpha = p4_wfn.nalpha()
    nbeta = p4_wfn.nbeta()

    if kwargs["casscf"] != None:
        frozen_core = vanilla_wfn.frzcpi().sum()
        frozen_virtual = vanilla_wfn.frzvpi().sum()
    else:
        frozen_core = p4_wfn.frzcpi().sum()
        frozen_virtual = p4_wfn.frzvpi().sum()

    # Get orbital symmetry information and construct hf reference
    orbitals = []

    if kwargs["casscf"] != None:
        for irrep, block in enumerate(p4_wfn.epsilon_a_subset("MO", "ALL").nph):
            for orbital in block:
                orbitals.append([orbital, irrep])
    else:
        for irrep, block in enumerate(p4_wfn.epsilon_a_subset("MO", "ACTIVE").nph):
            for orbital in block:
                orbitals.append([orbital, irrep])
    orbitals.sort()

    if kwargs["casscf"] != None:
        occ_alpha_per_irrep = vanilla_wfn.occupation_a().nph
        occ_beta_per_irrep = vanilla_wfn.occupation_b().nph
    else:
        occ_alpha_per_irrep = p4_wfn.occupation_a().nph
        occ_beta_per_irrep = p4_wfn.occupation_b().nph

    if kwargs["casscf"] != None:
        count_per_irrep = list(vanilla_wfn.frzcpi().to_tuple())
    else:
        count_per_irrep = list(p4_wfn.frzcpi().to_tuple())

    hf_reference = []
    hf_orbital_energies = []
    orb_irreps_to_int = []

    for [orbital_energy, irrep] in orbitals:
        hf_reference.append(int(occ_alpha_per_irrep[irrep][count_per_irrep[irrep]]))
        hf_reference.append(int(occ_beta_per_irrep[irrep][count_per_irrep[irrep]]))
        count_per_irrep[irrep] += 1
        hf_orbital_energies.append(orbital_energy)
        orb_irreps_to_int.append(irrep)
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

    # Build second quantized Hamiltonian
    Hsq = qforte.SQOperator()
    Hsq.add(p4_Enuc_ref + frozen_core_energy, [], [])
    for i in range(frozen_core, nmo - frozen_virtual):
        ia = (i - frozen_core) * 2
        ib = (i - frozen_core) * 2 + 1
        for j in range(frozen_core, nmo - frozen_virtual):
            ja = (j - frozen_core) * 2
            jb = (j - frozen_core) * 2 + 1

            irrep = (
                orb_irreps_to_int[i - frozen_core] ^ orb_irreps_to_int[j - frozen_core]
            )
            if irrep == 0:
                Hsq.add(mo_oeis[i, j], [ia], [ja])
                Hsq.add(mo_oeis[i, j], [ib], [jb])

            for k in range(frozen_core, nmo - frozen_virtual):
                ka = (k - frozen_core) * 2
                kb = (k - frozen_core) * 2 + 1
                for l in range(frozen_core, nmo - frozen_virtual):
                    la = (l - frozen_core) * 2
                    lb = (l - frozen_core) * 2 + 1
                    irrep = 0
                    for idx in [i, j, k, l]:
                        irrep ^= orb_irreps_to_int[idx - frozen_core]
                    if irrep == 0:
                        if ia != jb and kb != la:
                            Hsq.add(mo_teis[i, l, k, j] / 2, [ia, jb], [kb, la])  # abba
                        if ib != ja and ka != lb:
                            Hsq.add(mo_teis[i, l, k, j] / 2, [ib, ja], [ka, lb])  # baab

                        if ia != ja and ka != la:
                            Hsq.add(mo_teis[i, l, k, j] / 2, [ia, ja], [ka, la])  # aaaa
                        if ib != jb and kb != lb:
                            Hsq.add(mo_teis[i, l, k, j] / 2, [ib, jb], [kb, lb])  # bbbb

    if dipole == True:
        mo_dipints = np.asarray(mints.ao_dipole())
        mo_dipints = [np.einsum("uj,vi,uv", C, C, mo_dipints[i]) for i in range(3)]

        frozen_core_dipole = [p4_dip_nuc[i] for i in range(3)]

        if frozen_core > 0:
            for i in range(frozen_core):
                for j in range(3):
                    frozen_core_dipole[j] += 2 * mo_dipints[j][i, i]

        # Build second quantized dipole moment operators (Mux, Muy, Muz)

        Musqs = []
        for axis in range(3):
            Musq = qforte.SQOperator()

            Musq.add(frozen_core_dipole[axis], [], [])

            for i in range(frozen_core, nmo - frozen_virtual):
                ia = (i - frozen_core) * 2
                ib = (i - frozen_core) * 2 + 1
                for j in range(frozen_core, nmo - frozen_virtual):
                    ja = (j - frozen_core) * 2
                    jb = (j - frozen_core) * 2 + 1
                    Musq.add(mo_dipints[axis][i, j], [ia], [ja])
                    Musq.add(mo_dipints[axis][i, j], [ib], [jb])
            Musqs.append(Musq)

    else:
        Musqs = None

    # Set attributes
    qforte_mol.nuclear_repulsion_energy = p4_Enuc_ref
    qforte_mol.hf_energy = p4_Escf
    qforte_mol.hf_reference = hf_reference
    qforte_mol.sq_hamiltonian = Hsq
    if Musqs != None:
        qforte_mol.sq_dipole_x = Musqs[0]
        qforte_mol.sq_dipole_y = Musqs[1]
        qforte_mol.sq_dipole_z = Musqs[2]
        qforte_mol.dipole_x = Musqs[0].jw_transform()
        qforte_mol.dipole_y = Musqs[1].jw_transform()
        qforte_mol.dipole_z = Musqs[2].jw_transform()

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

    # Dump everything to a JSON file to be loaded later if desired.
    if json_dump != False:
        norbs = nmo - frozen_virtual - frozen_core

        external_data = {}

        external_data["hf_reference"] = {}
        external_data["hf_reference"]["data"] = hf_reference
        external_data["hf_reference"][
            "description"
        ] = "Occupations of the different spin-orbitals"
        external_data["scalar_energy"] = {}
        external_data["scalar_energy"]["data"] = p4_Enuc_ref + frozen_core_energy
        external_data["scalar_energy"][
            "description"
        ] = "scalar energy (sum of nuclear repulsion and frozen core energy"

        external_data["oei"] = {}
        external_data["oei"]["data"] = []
        for p in range(norbs):
            pa = 2 * p
            pb = 2 * p + 1
            for q in range(norbs):
                qa = 2 * q
                qb = 2 * q + 1
                irrep = (
                    orb_irreps_to_int[i - frozen_core]
                    ^ orb_irreps_to_int[j - frozen_core]
                )
                if irrep == 0:
                    external_data["oei"]["data"].append(
                        (pa, qa, mo_oeis[p + frozen_core, q + frozen_core])
                    )
                    external_data["oei"]["data"].append(
                        (pb, qb, mo_oeis[p + frozen_core, q + frozen_core])
                    )
        external_data["oei"][
            "description"
        ] = "one-electron integrals as a list of tuples (i,j,<i|h|j>)"

        external_data["tei"] = {}
        external_data["tei"]["data"] = []
        for p in range(norbs):
            pa = 2 * p
            pb = 2 * p + 1
            for q in range(norbs):
                qa = 2 * q
                qb = 2 * q + 1
                for r in range(norbs):
                    ra = 2 * r
                    rb = 2 * r + 1
                    for s in range(norbs):
                        irrep = 0
                        for idx in [p, q, r, s]:
                            irrep ^= orb_irreps_to_int[idx]
                        if irrep == 0:
                            sa = 2 * s
                            sb = 2 * s + 1
                            # prqs = <pq|rs> = (pr|qs)
                            # (Spatial orbitals - Psi4 uses chemist's notation.)
                            prqs = mo_teis[
                                p + frozen_core,
                                r + frozen_core,
                                q + frozen_core,
                                s + frozen_core,
                            ]
                            psqr = mo_teis[
                                p + frozen_core,
                                s + frozen_core,
                                q + frozen_core,
                                r + frozen_core,
                            ]
                            # external_data['tei']['data'][p, q, r, s] = <pq||rs> = (pr||qs)
                            # (Spin-orbitals - QForte uses physicist's notation.)
                            external_data["tei"]["data"].append(
                                (pa, qa, ra, sa, prqs - psqr)
                            )
                            external_data["tei"]["data"].append((pa, qb, ra, sb, prqs))
                            external_data["tei"]["data"].append(
                                (pb, qa, ra, sb, -1 * psqr)
                            )
                            external_data["tei"]["data"].append(
                                (pa, qb, rb, sa, -1 * psqr)
                            )
                            external_data["tei"]["data"].append((pb, qa, rb, sa, prqs))
                            external_data["tei"]["data"].append(
                                (pb, qb, rb, sb, prqs - psqr)
                            )
        external_data["tei"][
            "description"
        ] = "antisymmetrized two-electron integrals as a list of tuples (i,j,k,l,<ij||kl>)"

        if dipole == True:
            external_data["frozen_dip"] = {}

            external_data["frozen_dip"]["data"] = frozen_core_dipole
            external_data["frozen_dip"][
                "description"
            ] = "nuclear and frozen-core contribution to dipole moment ([x, y, z] list)"

            external_data["dip_ints_x"] = {}
            external_data["dip_ints_y"] = {}
            external_data["dip_ints_z"] = {}
            external_data["dip_ints_x"]["data"] = []
            external_data["dip_ints_y"]["data"] = []
            external_data["dip_ints_z"]["data"] = []
            for p in range(norbs):
                pa = 2 * p
                pb = 2 * p + 1
                for q in range(norbs):
                    qa = 2 * q
                    qb = 2 * q + 1
                    external_data["dip_ints_x"]["data"].append(
                        (pa, qa, mo_dipints[0][p + frozen_core, q + frozen_core])
                    )
                    external_data["dip_ints_x"]["data"].append(
                        (pb, qb, mo_dipints[0][p + frozen_core, q + frozen_core])
                    )
                    external_data["dip_ints_y"]["data"].append(
                        (pa, qa, mo_dipints[1][p + frozen_core, q + frozen_core])
                    )
                    external_data["dip_ints_y"]["data"].append(
                        (pb, qb, mo_dipints[1][p + frozen_core, q + frozen_core])
                    )
                    external_data["dip_ints_z"]["data"].append(
                        (pa, qa, mo_dipints[2][p + frozen_core, q + frozen_core])
                    )
                    external_data["dip_ints_z"]["data"].append(
                        (pb, qb, mo_dipints[2][p + frozen_core, q + frozen_core])
                    )
            external_data["dip_ints_x"][
                "description"
            ] = "x dipole integrals for the active space"
            external_data["dip_ints_y"][
                "description"
            ] = "y dipole integrals for the active space"
            external_data["dip_ints_z"][
                "description"
            ] = "z dipole integrals for the active space"

        external_data["nso"] = {}
        external_data["nso"]["data"] = 2 * norbs
        external_data["nso"]["description"] = "number of active spin orbitals"

        external_data["na"] = {}
        external_data["na"]["data"] = nalpha - frozen_core
        external_data["na"]["description"] = "number of active alpha electrons"

        external_data["nb"] = {}
        external_data["nb"]["data"] = nbeta - frozen_core
        external_data["nb"]["description"] = "number of active beta electrons"

        external_data["point_group"] = {}
        external_data["point_group"]["data"] = point_group
        external_data["point_group"]["description"] = "point group."

        spin_irreps = []
        for i in qforte_mol.orb_irreps_to_int:
            spin_irreps += [i, i]
        external_data["symmetry"] = {}
        external_data["symmetry"]["data"] = spin_irreps
        external_data["symmetry"]["description"] = "irreps of each spatial orbital"

        with open(json_dump, "w") as f:
            json.dump(external_data, f, indent=0)

    return qforte_mol


def create_external_mol(**kwargs):
    """Builds a qforte Molecule object from an external json file containing
    the one and two electron integrals and numbers of alpha/beta electrons.

    Returns
    -------
    Molecule
        The qforte Molecule object which holds the molecular information.
    """

    qforte_mol = Molecule(
        multiplicity=kwargs["multiplicity"],
        charge=kwargs["charge"],
        filename=kwargs["filename"],
    )

    # open json file
    with open(kwargs["filename"]) as f:
        external_data = json.load(f)

    # extract symmetry information if found
    try:
        point_group = external_data["point_group"]["data"]
        print(f"{point_group} symmetry specified.")

    except KeyError:
        "No point group in JSON file.  Using C1 symmetry."
        point_group = "C1"

    irreps = qforte.irreps_of_point_groups(point_group)
    qforte_mol.point_group = [point_group, irreps]

    qforte_mol.orb_irreps = []
    qforte_mol.orb_irreps_to_int = []

    # we need the irreps of the spatial orbitals, but the
    # json file provides the irreps of the spin-orbitals
    if point_group == "C1":
        qforte_mol.orb_irreps = ["A"] * int(external_data["nso"]["data"] / 2)
        qforte_mol.orb_irreps_to_int = [0] * int(external_data["nso"]["data"] / 2)
    else:
        for int_irrep in external_data["symmetry"]["data"][::2]:
            qforte_mol.orb_irreps_to_int.append(int_irrep)
            qforte_mol.orb_irreps.append(irreps[int_irrep])

    # build sq hamiltonian
    qforte_sq_hamiltonian = qforte.SQOperator()
    qforte_sq_hamiltonian.add(external_data["scalar_energy"]["data"], [], [])

    for p, q, h_pq in external_data["oei"]["data"]:
        qforte_sq_hamiltonian.add(h_pq, [p], [q])

    for p, q, r, s, h_pqrs in external_data["tei"]["data"]:
        qforte_sq_hamiltonian.add(
            h_pqrs / 4.0, [p, q], [s, r]
        )  # only works in C1 symmetry

    try:
        x_scalar, y_scalar, z_scalar = external_data["frozen_dip"]["data"]
        qforte_mol.sq_dipole_x = qforte.SQOperator()
        qforte_mol.sq_dipole_x.add(x_scalar, [], [])
        qforte_mol.sq_dipole_y = qforte.SQOperator()
        qforte_mol.sq_dipole_y.add(y_scalar, [], [])
        qforte_mol.sq_dipole_z = qforte.SQOperator()
        qforte_mol.sq_dipole_z.add(z_scalar, [], [])
        for p, q, mu_pq in external_data["dip_ints_x"]["data"]:
            qforte_mol.sq_dipole_x.add(mu_pq, [p], [q])
        for p, q, mu_pq in external_data["dip_ints_y"]["data"]:
            qforte_mol.sq_dipole_y.add(mu_pq, [p], [q])
        for p, q, mu_pq in external_data["dip_ints_z"]["data"]:
            qforte_mol.sq_dipole_z.add(mu_pq, [p], [q])

        qforte_mol.dipole_x = qforte_mol.sq_dipole_x.jw_transform()
        qforte_mol.dipole_y = qforte_mol.sq_dipole_y.jw_transform()
        qforte_mol.dipole_z = qforte_mol.sq_dipole_z.jw_transform()
        print("Dipole operators constructed successfully.")
    except:
        print("Dipole operators not constructed.")

    try:
        hf_reference = external_data["hf_reference"]["data"]
    except KeyError:
        hf_reference = [0] * external_data["nso"]["data"]
        for occ_alpha in range(external_data["na"]["data"]):
            hf_reference[occ_alpha * 2] = 1
        for occ_beta in range(external_data["nb"]["data"]):
            hf_reference[occ_beta * 2 + 1] = 1

    qforte_mol.hf_reference = hf_reference

    qforte_mol.sq_hamiltonian = qforte_sq_hamiltonian

    qforte_mol.hamiltonian = qforte_sq_hamiltonian.jw_transform()

    return qforte_mol
