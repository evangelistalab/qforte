This module implements Quantum Phase Estimation with a first-order Trotter approx. to the Hamiltonian.

Developers are not convinced the implementation is correct. Even for single-reference problems,
we observe readouts in the 0.3-0.7 range by modifying the t dependence. This behavior should not be
expected of a correct implementation of a non-Trotterized version. It is not clear if this is an
artifact of the Trotter approximation or a bug in the code.

If anybody is inclined to develop this, please begin with a non-Trotterized implementation.
