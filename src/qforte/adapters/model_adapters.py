import qforte as qf
from qforte.system.molecular_info import System


def create_TFIM(n: int, h: float, J: float):
    """Creates a 1D Transverse Field Ising Model hamiltonian with
    open boundary conditions, i.e., no interaction between the
    first and last spin sites.

    n: int
        Number of lattice sites

    h: float
        Strength of magnetic field

    j: float
        Interaction strength
    """

    TFIM = System()
    TFIM.hamiltonian = qf.QubitOperator()

    circuit = [(-h, f"Z_{i}") for i in range(n)]
    circuit += [(-J, f"X_{i} X_{i+1}") for i in range(n - 1)]

    for coeff, op_str in circuit:
        TFIM.hamiltonian.add(coeff, qf.build_circuit(op_str))

    TFIM.hf_reference = [0] * n

    return TFIM
