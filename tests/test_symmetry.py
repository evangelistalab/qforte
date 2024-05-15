from pytest import approx, mark, raises
from unittest.mock import patch
from io import StringIO
from qforte import (
    system_factory,
    char_table,
    irreps_of_point_groups,
    UCCNVQE,
    ADAPTVQE,
    UCCNPQE,
)


class TestPointGroupSymmetry:
    @mark.skip(reason="ambiguous test case")
    def test_symmetry_attributes(self):
        groups = ["c1", "c2", "cs", "ci", "d2", "c2h", "c2v", "d2h"]

        irreps = [
            ["A"],
            ["A", "B"],
            ["Ap", "App"],
            ["Ag", "Au"],
            ["A", "B1", "B2", "B3"],
            ["Ag", "Bg", "Au", "Bu"],
            ["A1", "A2", "B1", "B2"],
            ["Ag", "B1g", "B2g", "B3g", "Au", "B1u", "B2u", "B3u"],
        ]

        orb_irreps = [
            ["A", "A", "A", "A", "A", "A", "A"],
            ["A", "A", "A", "B", "B", "A", "A"],
            ["Ap", "Ap", "Ap", "App", "Ap", "Ap", "Ap"],
            ["Ag", "Ag", "Au", "Au", "Au", "Ag", "Au"],
            ["A", "A", "B1", "B2", "B3", "A", "B1"],
            ["Ag", "Ag", "Au", "Bu", "Bu", "Ag", "Au"],
            ["A1", "A1", "A1", "B1", "B2", "A1", "A1"],
            ["Ag", "Ag", "B1u", "B2u", "B3u", "Ag", "B1u"],
        ]

        orb_irreps_to_int = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 1],
            [0, 0, 1, 2, 3, 0, 1],
            [0, 0, 2, 3, 3, 0, 2],
            [0, 0, 0, 2, 3, 0, 0],
            [0, 0, 5, 6, 7, 0, 5],
        ]

        c1_char_tbl = "==========> C1 <==========\n\n" "      A    \n\n" "A     A    \n"

        c2_char_tbl = (
            "==========> C2 <==========\n\n"
            "      A    B    \n\n"
            "A     A    B    \n"
            "B     B    A    \n"
        )

        cs_char_tbl = (
            "==========> Cs <==========\n\n"
            "      Ap   App  \n\n"
            "Ap    Ap   App  \n"
            "App   App  Ap   \n"
        )

        ci_char_tbl = (
            "==========> Ci <==========\n\n"
            "      Ag   Au   \n\n"
            "Ag    Ag   Au   \n"
            "Au    Au   Ag   \n"
        )

        d2_char_tbl = (
            "==========> D2 <==========\n\n"
            "      A    B1   B2   B3   \n\n"
            "A     A    B1   B2   B3   \n"
            "B1    B1   A    B3   B2   \n"
            "B2    B2   B3   A    B1   \n"
            "B3    B3   B2   B1   A    \n"
        )

        c2h_char_tbl = (
            "==========> C2h <==========\n\n"
            "      Ag   Bg   Au   Bu   \n\n"
            "Ag    Ag   Bg   Au   Bu   \n"
            "Bg    Bg   Ag   Bu   Au   \n"
            "Au    Au   Bu   Ag   Bg   \n"
            "Bu    Bu   Au   Bg   Ag   \n"
        )

        c2v_char_tbl = (
            "==========> C2v <==========\n\n"
            "      A1   A2   B1   B2   \n\n"
            "A1    A1   A2   B1   B2   \n"
            "A2    A2   A1   B2   B1   \n"
            "B1    B1   B2   A1   A2   \n"
            "B2    B2   B1   A2   A1   \n"
        )

        d2h_char_tbl = (
            "==========> D2h <==========\n\n"
            "      Ag   B1g  B2g  B3g  Au   B1u  B2u  B3u  \n\n"
            "Ag    Ag   B1g  B2g  B3g  Au   B1u  B2u  B3u  \n"
            "B1g   B1g  Ag   B3g  B2g  B1u  Au   B3u  B2u  \n"
            "B2g   B2g  B3g  Ag   B1g  B2u  B3u  Au   B1u  \n"
            "B3g   B3g  B2g  B1g  Ag   B3u  B2u  B1u  Au   \n"
            "Au    Au   B1u  B2u  B3u  Ag   B1g  B2g  B3g  \n"
            "B1u   B1u  Au   B3u  B2u  B1g  Ag   B3g  B2g  \n"
            "B2u   B2u  B3u  Au   B1u  B2g  B3g  Ag   B1g  \n"
            "B3u   B3u  B2u  B1u  Au   B3g  B2g  B1g  Ag   \n"
        )

        char_tables = [
            c1_char_tbl,
            c2_char_tbl,
            cs_char_tbl,
            ci_char_tbl,
            d2_char_tbl,
            c2h_char_tbl,
            c2v_char_tbl,
            d2h_char_tbl,
        ]

        for count, group in enumerate(groups):
            mol = system_factory(
                system_type="molecule",
                build_type="psi4",
                basis="sto-3g",
                mol_geometry=[
                    ("O", (0.0, 0.0, 0)),
                    ("H", (0.0, 0, -1.5)),
                    ("H", (0.0, 0, 1.5)),
                ],
                symmetry=group,
            )

            assert mol.point_group == [group, irreps[count]]
            assert mol.orb_irreps == orb_irreps[count]
            assert mol.orb_irreps_to_int == orb_irreps_to_int[count]

            with patch("sys.stdout", new=StringIO()) as fake_out:
                char_table([group, irreps[count]])
                assert fake_out.getvalue() == char_tables[count]

    @mark.parametrize(
        "method, options",
        [
            (UCCNVQE, {"pool_type": "SD"}),
            (ADAPTVQE, {"pool_type": "SD", "avqe_thresh": 1.0e-3}),
            (UCCNPQE, {"pool_type": "SD"}),
        ],
    )
    def test_symmetry_ucc(self, method, options):
        groups = ["c1", "c2", "ci", "cs", "d2", "c2h", "c2v", "d2h"]

        for count, group in enumerate(groups):
            mol = system_factory(
                system_type="molecule",
                build_type="psi4",
                basis="cc-pVDZ",
                mol_geometry=[("He", (0, 0, 0))],
                symmetry=group,
            )

            alg = method(mol, irrep=0)

            alg.run(**options)

            Egs = alg.get_gs_energy()
            Efci = -2.887594831090935  # FCI

            t_ops = [24, 12, 12, 16, 6, 8, 10, 6]

            assert Egs == approx(Efci, abs=1.0e-10)

            assert len(alg._pool_obj) == t_ops[count]

    def test_point_group_to_irreps(self):
        groups = ["c1", "c2", "ci", "cs", "d2", "c2h", "c2v", "d2h"]
        irreps = [
            ["A"],
            ["A", "B"],
            ["Ag", "Au"],
            ["Ap", "App"],
            ["A", "B1", "B2", "B3"],
            ["Ag", "Bg", "Au", "Bu"],
            ["A1", "A2", "B1", "B2"],
            ["Ag", "B1g", "B2g", "B3g", "Au", "B1u", "B2u", "B3u"],
        ]

        for idx, group in enumerate(groups):
            assert irreps_of_point_groups(group) == irreps[idx]

        with raises(
            ValueError,
            match="The given point group is not supported. Choose one of:\nc1, c2, ci, cs, d2, c2h, c2v, d2h",
        ):
            irreps_of_point_groups("d3h")
