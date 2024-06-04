import qforte as qf
import numpy as np


def symmetry_check(
    n_qubits, qc, irreps, orb_irreps_to_int, target_N, target_Sz, target_irrep
):
    """
    This function performs a symmetry analysis of the provided quantum state.
    In particular, it computes the expectation values and variances of the
    particle number N, projection of total spin on z axis Sz, and total spin
    squared S^2 operators. In addition, it computes the weight contribution
    to the wavefunction of the various N, Sz, and spatial symmetry sectors.

    Arguments
    =========

    n_qubits: int
        The number of qubits of the system.

    qc: Computer
        The quantum computer prepared in the derised state.

    irreps: list of str
        List that holds the irreps of the group in the Cotton ordering.

    orb_irreps_to_int: list of int
        Integer n indexes the irrep of spatial orbital n.

    target_N: int
        The number of particles in the target quantum state.

    target_Sz: float
        The Sz eigenvalue of the target quantum state.

    target_irrep: int
        Integer representing the irrep of the target state
    """

    # Compute <N> and <N^2> - <N>^2, where N is the total number operator
    N = qf.QubitOperator()
    circ_const = qf.Circuit()
    N.add_term(n_qubits / 2, circ_const)
    for qubit in range(n_qubits):
        circ_Z = qf.Circuit()
        circ_Z.add_gate(qf.gate("Z", qubit))
        N.add_term(-0.5, circ_Z)
    N_sqrd = qf.QubitOperator()
    N_sqrd.add_op(N)
    N_sqrd.operator_product(N, True, True)
    N_exp_val = qc.direct_op_exp_val(N)
    N_var = qc.direct_op_exp_val(N_sqrd) - N_exp_val**2
    if np.imag(N_exp_val) != 0 or np.imag(N_var) != 0:
        raise ValueError(
            "The expectation value and variance of the particle number should be real!"
        )
    N_exp_val = np.real(N_exp_val)
    N_var = np.real(N_var)

    # Compute <Sz> and <Sz^2> - <Sz>^2, where Sz is the projection of the total spin on the z axis
    Sz = qf.total_spin_z(n_qubits)
    Sz_sqrd = qf.QubitOperator()
    Sz_sqrd.add_op(Sz)
    Sz_sqrd.operator_product(Sz, True, True)
    Sz_exp_val = qc.direct_op_exp_val(Sz)
    Sz_var = qc.direct_op_exp_val(Sz_sqrd) - Sz_exp_val**2
    if np.imag(Sz_exp_val) != 0 or np.imag(Sz_var) != 0:
        raise ValueError("The expectation value and variance of Sz should be real!")
    Sz_exp_val = np.real(Sz_exp_val)
    Sz_var = np.real(Sz_var)

    # Compute <S^2> and <S^4> - <S^2>^2, where S is the total spin
    S_sqrd = qf.total_spin_squared(n_qubits)
    S_sqrd_sqrd = qf.QubitOperator()
    S_sqrd_sqrd.add_op(S_sqrd)
    S_sqrd_sqrd.operator_product(S_sqrd, True, True)
    S_sqrd_exp_val = qc.direct_op_exp_val(S_sqrd)
    S_sqrd_var = qc.direct_op_exp_val(S_sqrd_sqrd) - S_sqrd_exp_val**2
    if np.imag(S_sqrd_exp_val) != 0 or np.imag(S_sqrd_var) != 0:
        raise ValueError("The expectation value and variance of S^2 should be real!")
    S_sqrd_exp_val = np.real(S_sqrd_exp_val)
    S_sqrd_var = np.real(S_sqrd_var)

    # Compute the weight of various symmetry subspaces in the wavefunction
    weight_target = 0
    weight_per_particle_number = [0] * (n_qubits + 1)
    weight_per_Sz = [0] * (n_qubits + 1)
    Sz_values = []
    Sz_values_dictionary = {}
    for i in range(n_qubits + 1):
        Sz_values.append(-n_qubits * 0.25 + 0.5 * i)
        Sz_values_dictionary[Sz_values[i]] = i
    weight_per_irrep = [0] * len(irreps)
    for det in range(1 << n_qubits):
        occ_alpha = []
        occ_beta = []
        for i in range(0, n_qubits, 2):
            if (1 << i) & det != 0:
                occ_alpha.append(i)
            if (1 << (i + 1)) & det != 0:
                occ_beta.append(i + 1)
        weight = qc.get_coeff_vec()[det]
        weight = np.real(weight * np.conjugate(weight))
        particle_number = len(occ_alpha + occ_beta)
        Sz_value = (len(occ_alpha) - len(occ_beta)) * 0.5
        irrep = qf.find_irrep(orb_irreps_to_int, occ_alpha + occ_beta)
        if (
            particle_number == target_N
            and Sz_value == target_Sz
            and irrep == target_irrep
        ):
            weight_target += weight
        weight_per_particle_number[particle_number] += weight
        weight_per_Sz[Sz_values_dictionary[Sz_value]] += weight
        weight_per_irrep[irrep] += weight

    print("\n\n*******************")
    print("*Symmetry Analysis*")
    print("*******************")

    print("\nParticle number symmetry analysis")
    print("=================================")

    for i in range(n_qubits + 1):
        print(
            "N =",
            str(i).rjust(2, " "),
            "weight:",
            f"{weight_per_particle_number[i]:.10f}",
        )

    print("<N>:                      ", f"{N_exp_val:+.10f}")
    print("<N^2> - <N>^2:            ", f"{N_var:+.10f}")

    print("\nSz symmetry analysis")
    print("====================")

    for i in range(n_qubits + 1):
        print(
            "Sz =",
            str(Sz_values[i]).rjust(4, " "),
            "weight:",
            f"{weight_per_Sz[i]:.10f}",
        )
    print("<Sz>:                     ", f"{Sz_exp_val:+.10f}")
    print("<Sz^2> - <Sz>^2:          ", f"{Sz_var:+.10f}")

    print("\nS^2 symmetry analysis")
    print("=====================")

    print("<S^2>:                    ", f"{S_sqrd_exp_val:+.10f}")
    print("<S^4> - <S^2>^2:          ", f"{S_sqrd_var:+.10f}")

    print("\nSpatial symmetry analysis")
    print("=========================")
    for i, j in enumerate(irreps):
        print(j.ljust(3, " "), "weight:", f"{weight_per_irrep[i]:.10f}")

    print("\nWeight of target symmetry subspace")
    print("==================================")
    print(
        "N = "
        + str(target_N)
        + ", Sz = "
        + str(target_Sz)
        + ", "
        + irreps[target_irrep]
        + " weight:",
        f"{weight_target:.10f}",
    )
