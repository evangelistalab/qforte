import subprocess
import os


class TestMain:
    def test_cpp(self):
        print("\n\nTesting C++ code...")
        subprocess.check_call(
            os.path.join(
                os.path.dirname(os.path.relpath(__file__)), "bin", "qforte_test"
            )
        )
        print()  # for prettier output
