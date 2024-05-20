import pytest
from pytest import approx
from qforte import UCCNVQE, UCCNPQE, ADAPTVQE, SPQE, system_factory


class TestEfficientCircuits:
    @pytest.mark.parametrize(
        "method, options",
        [
            (UCCNVQE, {"pool_type": "SD"}),
            (UCCNVQE, {"pool_type": "sa_SD"}),
            (ADAPTVQE, {"pool_type": "SD", "avqe_thresh": 1.0e-3}),
            (UCCNPQE, {"pool_type": "SD"}),
            (SPQE, {"spqe_thresh": 0.01}),
        ],
    )
    def test_feb_qeb_circuits(self, method, options):
        Rhh = 1.5

        mol = system_factory(
            system_type="molecule",
            build_type="psi4",
            basis="sto-6g",
            mol_geometry=[
                ("H", (0, 0, -3 * Rhh / 2)),
                ("H", (0, 0, -Rhh / 2)),
                ("H", (0, 0, Rhh / 2)),
                ("H", (0, 0, 3 * Rhh / 2)),
            ],
            symmetry="d2h",
            multiplicity=1,  # Only singlets will work with QForte
            charge=0,
            num_frozen_docc=0,
            num_frozen_uocc=0,
            run_mp2=0,
            run_ccsd=0,
            run_cisd=0,
            run_fci=0,
        )

        for qubit_excit in [False, True]:
            standard = method(
                mol, compact_excitations=False, qubit_excitations=qubit_excit
            )
            standard.run(**options)

            efficient = method(
                mol, compact_excitations=True, qubit_excitations=qubit_excit
            )
            efficient.run(**options)

            # Since the standard and efficient circuits are equivalent, they produce practically
            # identical energies
            assert standard.get_gs_energy() == approx(
                efficient.get_gs_energy(), abs=1.0e-10
            )
            # The standard circuits should contain a larger number of CNOTs compared to their
            # efficient counterparts
            assert standard._n_cnot > efficient._n_cnot

            if method == SPQE and qubit_excit == True:
                # Compute CNOT count analytically and compare with those coming out of SPQE
                analytic_standard_cnots = 0
                analytic_efficient_cnots = 0
                for rank_minus_one, nparams in enumerate(standard._nbody_counts):
                    analytic_standard_cnots += nparams * analytic_standard(
                        rank_minus_one + 1
                    )
                for rank_minus_one, nparams in enumerate(efficient._nbody_counts):
                    analytic_efficient_cnots += nparams * analytic_efficient(
                        rank_minus_one + 1
                    )

                assert analytic_standard_cnots == standard._n_cnot
                assert analytic_efficient_cnots == efficient._n_cnot


def analytic_standard(rank):
    cnots = (2 * rank - 1) * (1 << (2 * rank))
    return cnots


def analytic_efficient(rank):
    cnots = (1 << (2 * rank - 1)) + 4 * rank - 2
    return cnots
