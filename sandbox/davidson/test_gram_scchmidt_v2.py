import numpy as np
import qforte as qf

from qforte.maths import gram_schmidt as qfgs

print("I get here")

def gram_schmidt(vectors):
    result = []
    
    for v in vectors:
        u = np.array(v)
        
        for w in result:
            proj = np.dot(v, w) / np.dot(w, w)
            projection = proj * np.array(w)
            u = u - projection
        
        norm = np.linalg.norm(u)
        u /= norm

        result.append(u.tolist())

    return result

# Example: vectors in R^3
vectors = [
    [1., 2., 0.], 
    [2., -1., 1.], 
    [1., 1., -1.]
    ]

vectors_ary = np.asarray(vectors)

tensors = []

for v in vectors_ary:
    V = qf.Tensor(v.shape, "V")
    V.fill_from_nparray(v.ravel(), V.shape())
    print(V)
    tensors.append(V)

# Apply Gram-Schmidt in qforte 
result_qf = qfgs.orthogonalize(tensors)
print(f"\n\n===> result <==== from qforte gs")
for V in result_qf:
    print(V)

# Check ortogonality with overlap matrix
S = qf.Tensor([len(vectors), len(vectors)], "overalp")
for i, v1 in enumerate(result_qf):
    for j, v2 in enumerate(result_qf):
        S.set([i,j], v1.vector_dot(v2))
        
print(S)

# Apply Gram-Schmidt above
result_jg = gram_schmidt(vectors)
print(f"\n\n===> result <==== from josh goings")
for v in result_jg:
    print(v)

# Apply Gram-Schmidt in numpy QR code
print(f"\n\n===> result <==== from numpy QR decomp")
Q, R = np.linalg.qr(vectors_ary.T)

print(Q.T)

print()