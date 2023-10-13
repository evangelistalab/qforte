import qforte as qf

"""
Takes a list of vectors (qforte Tensor objects)
and orthogonalizes them NOT in place
Will be needed in the davidson algorithm,

Specifically a QR decomposition of a matrix A where the columnts of A are non-orrhogonal
will be orghoganalized by gramshchmidt and returned in Q
"""
def orthogonalize(vectors):
    result = []
    
    for V in vectors:
        U = qf.Tensor(V.shape(), 'U')
        U.copy_in(V)
        
        # Subtract projections on previous vectors
        for W in result:
            W2 = qf.Tensor(V.shape(), 'W2')
            W2.copy_in(W)

            # careful here, need to check how vector_dot product works... <V|W> vs <W|V>
            proj = V.vector_dot(W2) / (W2.norm()**2)
            W2.scale(proj)
            U.subtract(W2)

        norm = U.norm()
        U.scale(1.0/norm)
        result.append(U)

    return result