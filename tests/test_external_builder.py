from pytest import approx
from qforte import system_factory, UCCNPQE

import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(THIS_DIR, "as_ints.json")


class TestExternalBuilder:
    def test_H2_uccsd_pqe_exact(self):
        """
        This test checks the external builder using the json file found
        in the forte/tests/methods/external_solver-2/as_ints.json.
        """

        # The FCI energy of H2/cc-pVDZ (0.7 angs) in a two-two active space
        # can be found in forte/tests/methods/external_solver-2/output.ref
        E_fci = -1.129181390402433

        mol = system_factory(
            system_type="molecule", build_type="external", filename=data_path
        )

        alg = UCCNPQE(mol)
        alg.run(pool_type="SD", opt_thresh=1.0e-7)

        Egs_elec = alg.get_gs_energy()
        assert Egs_elec == approx(E_fci, abs=1.0e-12)
