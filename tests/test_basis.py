from qforte import QubitBasis


class TestBasis:
    def test_str(self):
        assert (
            str(QubitBasis(0))
            == "|0000000000000000000000000000000000000000000000000000000000000000>"
        )
        assert (
            str(QubitBasis(5))
            == "|1010000000000000000000000000000000000000000000000000000000000000>"
        )
