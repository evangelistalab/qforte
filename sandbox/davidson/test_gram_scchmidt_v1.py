import numpy as np
import qforte as qf

def gram_schmidt(vectors):
    result = []
    
    for v in vectors:
        u = np.array(v)
        
        # Subtract projections on previous vectors
        for w in result:
            proj = np.dot(v, w) / np.dot(w, w)
            projection = proj * np.array(w)
            u = u - projection
        
        # Normalize
        norm = np.linalg.norm(u)
        u /= norm

        result.append(u.tolist())

    result = np.asarray(result)

    return result

# Example: vectors in R^3
vectors = [[1., 2., 0.], [2., -1., 1.], [1., 1., -1.]]

vectors = np.asarray(vectors)

# Apply Gram-Schmidt
result = gram_schmidt(vectors)


print(result)

Q, R = np.linalg.qr(vectors.T)

print(Q.T)

print()