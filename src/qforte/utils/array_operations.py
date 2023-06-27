"""
array_operations.py
=================================================
A module for working with Scipy arrays
"""

import numpy as np
import scipy

def sq_op_to_scipy(sq_op, N_qubits):
    #Function to convert a second-quantized operator to a sparse, complex matrix
    #Draws heavily from the sparse_tools module in OpenFermion
    I = scipy.sparse.csc_matrix(np.array([[1+0j,0],[0,1+0j]]))
    X = scipy.sparse.csc_matrix(np.array([[0,1+0j],[1+0j,0]]))
    Y = scipy.sparse.csc_matrix(np.array([[0,0-1j],[0+1j,0]]))
    Z = scipy.sparse.csc_matrix(np.array([[1+0j,0],[0,-1+0j]]))

    #Hilbert space dimension
    dim = int(2**N_qubits)

    #Build all the annihilation operators
    annihilators = []
    zvec = np.ones((1))
    for i in range(0, N_qubits):
        ann = scipy.sparse.csc_matrix(.5*np.ones((1,1)))
        big_Z = scipy.sparse.diags(zvec) 
        big_id = scipy.sparse.identity(2**(N_qubits - i - 1), dtype = "complex", format = "csc")
        ann = scipy.sparse.kron(ann, big_id)
        ann = scipy.sparse.kron(ann, X + 1j*Y)
        ann = scipy.sparse.kron(ann, big_Z)     
        zvec = np.kron(zvec, np.array([1,-1]))
        ann.eliminate_zeros()
        annihilators.append(ann)
    
    #Convert each excitation string to a sparse matrix and add them up.
    terms = sq_op.terms()
    vals = [[]]
    rows = [[]]
    cols = [[]]
    for term in terms:
        coeff = term[0]
        term_mat = scipy.sparse.identity(dim, dtype = "complex", format = "csc")
        for i in term[1]:
            term_mat = annihilators[i]@term_mat
        for i in term[2]:
            term_mat = annihilators[i].getH()@term_mat
        term_mat = term_mat.tocoo(copy = False)
        term_mat.eliminate_zeros()
        vals.append(coeff * term_mat.data)
        row, col = term_mat.nonzero()
        rows.append(row)
        cols.append(col)    
    vals = np.concatenate(vals)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    arr = scipy.sparse.coo_matrix((vals, (rows, cols)), shape = (dim, dim))    
    arr = arr.tocsc(copy = False)    
    arr.eliminate_zeros() 
    return scipy.sparse.csc_matrix(arr)