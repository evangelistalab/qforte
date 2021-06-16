
Overview
========

.. sectionauthor:: Nicholas H. Stair

QForte is an open-source suite of state-of-the-art quantum algorithms library for
electronic structure. All algorithms rely on an in-house state vector simulator
written in C++ and exposed in Python via Pybind11.

Capabilities
------------

In general, QForte is composed several distinct algorithmic catagories, each
with their own more specific algorithm implementations:

#. Variational quantum eigensolvers (VQE)
    #. Trotterized (disentangled) Unitary coupled cluster VQE (UCC-VQE)
    #. Adaptive derivative assembled pseudo-Trotter VQE (ADAPT-VQE)

#. Projective quantum eigensolvers (PQE)
    #. Trotterized (disentangled) Unitary coupled cluster PQE (UCC-PQE)
    #. Selected PQE (SPQE)

#. Quantum subspace diagonalization (QSD)
    #. Quantum Krylov (QK)
    #. Multireference selected quantum Krylov (MRSQK)

#. Imagniary time evolution (ITE)
    #. Quantum imaginary time evolution (QITE)
    #. Quantum imaginary time Lanczos (QLanczos)

#. Quantum phase estimation (QPE)
    #. Canonical quantum phase estimation 


Dependencies
------------

In order to run Forte, the following are required:

#. A Recent version of Psi4
#. CMake version 3.0 or higher
