from pytest import approx
from qforte import ADAPTVQE
from qforte import system_factory
from qforte import sq_op_to_scipy
from qforte import ritz_eigh
from qforte import cisd_manifold
from qforte import build_refprep
from qforte import build_effective_array
from qforte import build_effective_symmetric_operator
from qforte import compute_operator_matrix_element
from qforte import Computer
from qforte import Circuit
from qforte import sa_single
import copy
import os
import numpy as np
import psi4

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(THIS_DIR, "lih_cas_dump.json")
if os.path.exists(data_path):
    os.remove(data_path)


class TestSAADAPTVQE:
    def test_LiH_more_adapt_vqe(self):
        geom = [("Li", (0, 0, 0)), ("H", (0, 0, 1))]
        mol = system_factory(
            system_type="molecule",
            mol_geometry=geom,
            build_type="psi4",
            basis="sto-3g",
            dipole=True,
            num_frozen_docc=1,
            num_frozen_uocc=1,
            symmetry="C2v",
        )

        refs = [mol.hf_reference] + cisd_manifold(mol.hf_reference)
        weights = [2 ** (-i - 1) for i in range(len(refs))]
        weights[-1] += 2 ** (-len(weights))

        alg = ADAPTVQE(
            mol,
            print_summary_file=False,
            is_multi_state=True,
            reference=refs,
            weights=weights,
            compact_excitations=True,
        )

        H = sq_op_to_scipy(mol.sq_hamiltonian, alg._nqb, Sz=0, N=2).todense()

        w, v = np.linalg.eigh(H)

        non_degens = [0, 1, 2, 7, 8, 15]
        w = w[non_degens]
        v = v[:, non_degens]

        dip_x_arr = sq_op_to_scipy(mol.sq_dipole_x, alg._nqb).todense()
        dip_y_arr = sq_op_to_scipy(mol.sq_dipole_y, alg._nqb).todense()
        dip_z_arr = sq_op_to_scipy(mol.sq_dipole_z, alg._nqb).todense()

        alg.run(
            avqe_thresh=1e-12,
            pool_type="GSD",
            opt_thresh=1e-10,
            opt_maxiter=1000,
            adapt_maxiter=1,
        )

        U = alg.build_Uvqc(amplitudes=alg._tamps)

        Es, A, ops = ritz_eigh(
            alg._nqb, mol.hamiltonian, U, [mol.dipole_x, mol.dipole_y, mol.dipole_z]
        )
        dip_x, dip_y, dip_z = ops

        Es = Es[non_degens]

        for i in range(len(Es)):
            assert Es[i] == approx(w[i], abs=1.0e-10)

        total_dip = np.zeros(dip_x.shape)
        for op in [dip_x, dip_y, dip_z]:
            total_dip += np.multiply(op.conj(), op).real
        total_dip = np.sqrt(total_dip)
        total_dip = total_dip[np.ix_(non_degens, non_degens)]

        dip_dir = np.zeros((len(Es), len(Es)))
        for i in range(len(Es)):
            for op in [dip_x_arr, dip_y_arr, dip_z_arr]:
                sig = op @ v[:, i]
                for j in range(len(Es)):
                    dip_dir[i, j] += (
                        (sig.T.conj() @ v[:, j])[0, 0] * (v[:, j].T.conj() @ sig)[0, 0]
                    ).real
        dip_dir = np.sqrt(dip_dir)

        for i in range(len(Es)):
            for j in range(len(Es)):
                assert dip_dir[i, j] - total_dip[i, j] == approx(0.0, abs=1e-10)

        circ_refs = [build_refprep(ref) for ref in refs]

        alg = ADAPTVQE(
            mol,
            print_summary_file=False,
            is_multi_state=True,
            reference=circ_refs,
            weights=weights,
            compact_excitations=True,
            state_prep_type="unitary_circ",
        )

        alg.run(
            avqe_thresh=1e-12,
            pool_type="GSD",
            opt_thresh=1e-10,
            opt_maxiter=1000,
            adapt_maxiter=1,
        )

        U = alg.build_Uvqc(amplitudes=alg._tamps)

        Es, A, ops = ritz_eigh(
            alg._nqb, mol.hamiltonian, U, [mol.dipole_x, mol.dipole_y, mol.dipole_z]
        )
        dip_x, dip_y, dip_z = ops
        Es = Es[non_degens]

        for i in range(len(Es)):
            assert Es[i] == approx(w[i], abs=1.0e-10)

        total_dip = np.zeros(dip_x.shape)
        for op in [dip_x, dip_y, dip_z]:
            total_dip += np.multiply(op.conj(), op).real
        total_dip = np.sqrt(total_dip)
        total_dip = total_dip[np.ix_(non_degens, non_degens)]

        for i in range(len(Es)):
            for j in range(len(Es)):
                assert dip_dir[i, j] - total_dip[i, j] == approx(0.0, abs=1e-10)

        alg = ADAPTVQE(
            mol,
            print_summary_file=False,
            is_multi_state=True,
            reference=refs,
            weights=weights,
            compact_excitations=False,
        )

        alg.run(
            avqe_thresh=1e-12,
            pool_type="GSD",
            opt_thresh=1e-10,
            opt_maxiter=1000,
            adapt_maxiter=1,
            tamps=[],
            tops=[],
        )

        U = alg.build_Uvqc(amplitudes=alg._tamps)

        Es, A, ops = ritz_eigh(
            alg._nqb, mol.hamiltonian, U, [mol.dipole_x, mol.dipole_y, mol.dipole_z]
        )
        dip_x, dip_y, dip_z = ops

        Es = Es[non_degens]
        for i in range(len(Es)):
            assert Es[i] == approx(w[i], abs=1.0e-10)

        total_dip = np.zeros(dip_x.shape)
        for op in [dip_x, dip_y, dip_z]:
            total_dip += np.multiply(op.conj(), op).real
        total_dip = np.sqrt(total_dip)
        total_dip = total_dip[np.ix_(non_degens, non_degens)]

        for i in range(len(Es)):
            for j in range(len(Es)):
                assert dip_dir[i, j] - total_dip[i, j] == approx(0.0, abs=1e-10)

        refs = [np.zeros((256)) for i in range(2)]
        refs[0][192] = 1.0
        refs[1][48] = 1.0
        computers = [Computer(8), Computer(8)]
        computers[0].set_coeff_vec(refs[0])
        computers[1].set_coeff_vec(refs[1])
        weights = [0.6, 0.4]

        alg = ADAPTVQE(
            mol,
            print_summary_file=False,
            is_multi_state=True,
            reference=computers,
            weights=weights,
            compact_excitations=False,
            state_prep_type="computer",
        )

        alg.run(
            avqe_thresh=1e-12,
            pool_type="GSD",
            opt_thresh=1e-10,
            opt_maxiter=1000,
            adapt_maxiter=30,
            tamps=[],
            tops=[],
        )

        Uvqc = alg.build_Uvqc(amplitudes=alg._tamps)[0]
        H_eff = build_effective_array(
            mol.hamiltonian, Uvqc, alg.get_initial_computer()
        ).real
        dip_x_eff = build_effective_array(
            mol.dipole_x, Uvqc, alg.get_initial_computer()
        ).real
        dip_y_eff = build_effective_array(
            mol.dipole_y, Uvqc, alg.get_initial_computer()
        ).real
        dip_z_eff = build_effective_array(
            mol.dipole_z, Uvqc, alg.get_initial_computer()
        ).real

        E_more, C_more = np.linalg.eigh(H_eff)
        dip_x_more = C_more.T @ dip_x_eff @ C_more
        dip_y_more = C_more.T @ dip_y_eff @ C_more
        dip_z_more = C_more.T @ dip_z_eff @ C_more

        for i in range(len(E_more)):
            assert E_more[i] == approx(w[i], abs=1.0e-10)

        total_dip = np.zeros(dip_x_more.shape)
        for op in [dip_x_more, dip_y_more, dip_z_more]:
            total_dip += np.multiply(op, op)
        total_dip = np.sqrt(total_dip)

        for i in range(len(E_more)):
            for j in range(len(E_more)):
                assert dip_dir[i, j] - total_dip[i, j] == approx(0.0, abs=1e-7)

        spaces = [[1, 0, 0, 0], [2, 0, 0, 0], [1, 0, 1, 1]]
        mol = system_factory(
            system_type="molecule",
            mol_geometry=geom,
            build_type="psi4",
            basis="sto-6g",
            dipole=True,
            symmetry="C2v",
            casscf=spaces,
            no_com=True,
            no_reorient=True,
            json_dump=data_path,
        )

        mol = system_factory(build_type="external", filename=data_path)

        occ_refs = [
            [1, 1, 1, 1, 0, 0] + [0] * 6,
            [1, 1, 0, 0, 1, 1] + [0] * 6,
            [1, 1, 0, 1, 1, 0] + [0] * 6,
            [1, 1, 1, 0, 0, 1] + [0] * 6,
        ]

        alg = ADAPTVQE(
            mol,
            print_summary_file=False,
            is_multi_state=True,
            reference=occ_refs,
            weights=[0.25] * 4,
            compact_excitations=True,
        )

        H_eff = build_effective_symmetric_operator(
            12, mol.hamiltonian, alg._refprep
        ).real
        E_casscf = np.linalg.eigh(H_eff)[0][0]
        assert E_casscf == approx(-7.873605319132174, 1e-8)

        alg.run(pool_type="GSD", adapt_maxiter=3)

        correct_Es = [
            -7.8593451680521662,
            -7.7158474059591828,
            -7.6836465355578119,
            -7.2065322341909379,
        ]

        for i in range(4):
            assert correct_Es[i] == approx(alg._diag_energies[-1][i])

        comp_refs = [Computer(12) for i in range(4)]

        coeff_vec = np.zeros(2**12)
        coeff_vec[int("001111", 2)] = 1
        comp_refs[0].set_coeff_vec(copy.deepcopy(coeff_vec))

        coeff_vec = np.zeros(2**12)
        coeff_vec[int("110011", 2)] = 1
        comp_refs[1].set_coeff_vec(copy.deepcopy(coeff_vec))

        coeff_vec = np.zeros(2**12)
        coeff_vec[int("100111", 2)] = 1
        comp_refs[2].set_coeff_vec(copy.deepcopy(coeff_vec))

        coeff_vec = np.zeros(2**12)
        coeff_vec[int("011011", 2)] = 1
        comp_refs[3].set_coeff_vec(copy.deepcopy(coeff_vec))

        alg = ADAPTVQE(
            mol,
            print_summary_file=False,
            is_multi_state=True,
            reference=comp_refs,
            weights=[0.25] * 4,
            compact_excitations=True,
            state_prep_type="computer",
        )

        alg.run(pool_type="GSD", adapt_maxiter=3)
        for i in range(4):
            assert correct_Es[i] == approx(alg._diag_energies[-1][i])

        circ_refs = []

        circ_refs.append(build_refprep([1] * 2 + [1, 1, 0, 0] + [0] * 6))
        circ_refs.append(build_refprep([1] * 2 + [0, 0, 1, 1] + [0] * 6))
        circ_refs.append(build_refprep([1] * 2 + [0, 1, 1, 0] + [0] * 6))
        circ_refs.append(build_refprep([1] * 2 + [1, 0, 0, 1] + [0] * 6))

        alg = ADAPTVQE(
            mol,
            print_summary_file=False,
            is_multi_state=True,
            reference=circ_refs,
            weights=[0.25] * 4,
            compact_excitations=True,
            state_prep_type="unitary_circ",
        )

        alg.run(pool_type="GSD", adapt_maxiter=3)
        for i in range(4):
            assert correct_Es[i] == approx(alg._diag_energies[-1][i])

        psi4.core.clean_options()
