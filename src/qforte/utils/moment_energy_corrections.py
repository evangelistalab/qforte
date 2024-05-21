"""
Functions for computing non-iterative energy corrections based on the
method of moments of coupled-cluster (MMCC) theory. The orginal ideas
for traditional coupled-cluster approaches can be found in:
    DOI: 10.1007/BF01117411
    DOI: 10.1142/9789812792501_0001
    DOI: 10.1063/1.481769
    DOI: 10.1063/1.2137318
The use of such moment corrections in the UCC case is experimental
and preliminary data is published in 10.1021/acs.jpca.3c02781
"""

import qforte as qf
from qforte.utils.trotterization import trotterize
from qforte import find_irrep
import numpy as np


def construct_moment_space(self):
    # List that will hold the orbital energy differences used in Moller-Plesset denominators
    self._mpdenom = []
    # List that will hold <Phi_k| H |Phi_k> used in the Epstein-Nesbet denominators
    self._epstein_nesbet = []

    self._E_mmcc_mp = []
    self._E_mmcc_en = []

    # List that will store the integers representing the determinants used in the moment corrections
    self._mmcc_excitation_indices = []
    self._mmcc_pool = qf.SQOpPool()

    if not hasattr(self, "_orb_e"):
        self.build_orb_energies()

    # create a pool of particle number, Sz, and spatial symmetry adapted second quantized operators
    # Encode the occupation list into a bitstring
    ref = sum([b << i for i, b in enumerate(self._ref)])
    # `& mask_alpha` gives the alpha component of a bitstring. `& mask_beta` does likewise.
    mask_alpha = 0x5555555555555555
    mask_beta = mask_alpha << 1
    nalpha = sum(self._ref[0::2])
    nbeta = sum(self._ref[1::2])

    if not isinstance(self._max_moment_rank, int) or self._max_moment_rank < 0:
        raise TypeError("The maximum moment rank must be a non-negative integer!")
    elif self._max_moment_rank > nalpha + nbeta:
        # WARNING: This step could be potetnially removed, if QForte allows for methods
        #          that violate particle number symmetry
        self._max_moment_rank = nalpha + nbeta

    for I in range(1 << self._nqb):
        alphas = [int(j) for j in bin(I & mask_alpha)[2:]]
        betas = [int(j) for j in bin(I & mask_beta)[2:]]
        # The following checks could be potentially removed, if QForte gets methods
        # that violate particle number, Sz, and spatial symmetries
        if sum(alphas) == nalpha and sum(betas) == nbeta:
            if (
                find_irrep(
                    self._sys.orb_irreps_to_int,
                    [len(alphas) - i - 1 for i, x in enumerate(alphas) if x]
                    + [len(betas) - i - 1 for i, x in enumerate(betas) if x],
                )
                == 0
            ):
                # Create the bitstring of created/annihilated orbitals
                excit = bin(ref ^ I).replace("0b", "")
                # Confirm excitation number is non-zero
                if excit != "0":
                    # Consider moments with rank <= self._max_moment_rank
                    if int(excit.count("1") / 2) <= self._max_moment_rank:
                        occ_idx = [
                            int(i)
                            for i, j in enumerate(reversed(excit))
                            if int(j) == 1 and self._ref[i] == 1
                        ]
                        unocc_idx = [
                            int(i)
                            for i, j in enumerate(reversed(excit))
                            if int(j) == 1 and self._ref[i] == 0
                        ]
                        sq_op = qf.SQOperator()
                        sq_op.add(+1.0, unocc_idx, occ_idx)
                        sq_op.add(-1.0, occ_idx[::-1], unocc_idx[::-1])
                        sq_op.simplify()
                        self._mmcc_pool.add_term(0.0, sq_op)
                        self._mpdenom.append(
                            sum(self._orb_e[x] for x in occ_idx)
                            - sum(self._orb_e[x] for x in unocc_idx)
                        )
                        self._mmcc_excitation_indices.append(I)

    for i in self._mmcc_pool.terms():
        sq_op = i[1]
        qc = qf.Computer(self._nqb)
        qc.apply_circuit(self._refprep)
        # This could be replaced by set_bit?
        qc.apply_operator(sq_op.jw_transform(self._qubit_excitations))
        self._epstein_nesbet.append(qc.direct_op_exp_val(self._qb_ham))


def compute_moment_energies(self):
    # The moments, i.e., residuals, are estimated by measuring the "residual" state a la SPQE

    if self._moment_dt is None:
        print(
            "WARNING: No value has been provided for the moment_dt variable required by the moment correction code!"
            "\nProceeding with the default value of 0.001"
        )
        self._moment_dt = 0.001

    _eiH, _ = trotterize(
        self._qb_ham,
        factor=self._moment_dt * (0.0 + 1.0j),
        trotter_number=self._trotter_number,
    )

    # do U^dag e^iH U |Phi_o> = |Phi_res>
    U = self.ansatz_circuit()

    qc_res = qf.Computer(self._nqb)
    qc_res.apply_circuit(self._Uprep)
    qc_res.apply_circuit(U)
    qc_res.apply_circuit(_eiH)
    qc_res.apply_circuit(U.adjoint())

    res_coeffs = [i / self._moment_dt for i in qc_res.get_coeff_vec()]

    mmcc_res = [res_coeffs[I] for I in self._mmcc_excitation_indices]
    mmcc_res_sq_over_mpdenom = [
        np.real(np.conj(mmcc_res[I]) * mmcc_res[I] / self._mpdenom[I])
        for I in range(len(mmcc_res))
    ]
    self._E_mmcc_mp.append(self._curr_energy + sum(mmcc_res_sq_over_mpdenom))
    mmcc_res_sq_over_epstein_nesbet_denom = [
        np.real(
            np.conj(mmcc_res[I])
            * mmcc_res[I]
            / (self._curr_energy - self._epstein_nesbet[I])
        )
        for I in range(len(mmcc_res))
    ]
    self._E_mmcc_en.append(
        self._curr_energy + sum(mmcc_res_sq_over_epstein_nesbet_denom)
    )
