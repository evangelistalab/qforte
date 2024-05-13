from __future__ import division
from __future__ import print_function
import math
import numpy as np
import time

''' Block Davidson, Joshua Goings (2013)

    Block Davidson method for finding the first few
	lowest eigenvalues of a large, diagonally dominant,
    sparse Hermitian matrix (e.g. Hamiltonian)
'''

n = 10000 # was 1200				# Dimension of CI matrix (CI vector will be this size)
tol = 1e-8				# Convergence tolerance
mmax = n//2				# Maximum number of Davidson iterations	

''' Create sparse, diagonally dominant matrix A with 
	diagonal containing 1,2,3,...n. The eigenvalues
    should be very close to these values. You can 
    change the sparsity. A smaller number for sparsity
    increases the diagonal dominance. Larger values
    (e.g. sparsity = 1) create a dense matrix
'''

sparsity = 0.0001
A = np.zeros((n,n))
for i in range(0,n):
    A[i,i] = i + 1 
A = A + sparsity*np.random.randn(n,n) 
A = (A.T + A)/2 # the stand in for the hamiltonain matrix


k = 4 # was 8					# number of initial guess vectors 
eig = 3 					# number of eignvalues to solve 
t = np.eye(n,k)			# set of k unit vectors as guess
V = np.zeros((n,n))		# array of zeros to hold guess vec
I = np.eye(n)			# identity matrix same dimen as A

# Begin block Davidson routine

start_davidson = time.time()

print(f"number of elements in 'CI' vec n:  {n}")
print(f"max number of davidson iters mmax: {mmax}")
print(f"number of initial guess vectors k: {k}")
print(f"number of eignvalues to solve eig: {eig}")

# these are the macro iterations, you will add k expansion vectors at each iteration
# loop from k to mmax in multiples of k
for m in range(k,mmax,k):
    
    if m <= k:
        # loop over tial vectors on first iteration
        for j in range(0,k):
            V[:,j] = t[:,j]/np.linalg.norm(t[:,j])

        theta_old = 1 
    elif m > k:
        # more often
        theta_old = theta[:eig]


    # Perform a QR decompositon and set V[:, :m] = Q
    Q, R = np.linalg.qr(V[:,:m]) # use gram-schmidt to build Q

    V[:,:m] = Q
    print(f'  V[:,:m].shape {V[:,:m].shape}')


    # T is the subspace matrix we will diagonalize
    T = np.dot(V[:,:m].T, np.dot(A,V[:,:m]))
    
    print(f'  T.shape {T.shape}')
    # print(f'approximate eigenvalues T: {T}')

    # THETA are the subspace eigenvalues
    # S are the subspace eigenvectors, elements
    # become expansion coeffs for the expansion vector
    THETA, S = np.linalg.eig(T)

    # need idx to sort the eigenvalues of T
    idx = THETA.argsort()
    print(f'  idx {idx}')

    # the sorted lowest eigenvalues of T
    theta = THETA[idx]
    print(f'  theta {theta}')

    # the corresponding expansion coefficeints for each eigen-index idx
    s = S[:,idx]
    print(f'  T.shape {T.shape}')
    print(f'  s.shape {s.shape}')

    # loop j over the k trial vectors
    for j in range(0,k):
        print(f' s_itr.shape {s[:,j].shape}')
        w = np.dot(
            (A - theta[j]*I), np.dot(V[:,:m], s[:,j])
            ) 
        print(f'    w.shape {w.shape}')

        # the new j'th trial 
        q = w/(theta[j]-A[j,j])
        print(f'    q.shape {q.shape}')

        V[:,(m+j)] = q
        print(f'    V[:,(m+j)].shape {V[:,(m+j)].shape}')

    norm = np.linalg.norm(theta[:eig] - theta_old)
    if norm < tol:
        break

end_davidson = time.time()

# End of block Davidson. Print results.

print("davidson = ", theta[:eig],";",
    end_davidson - start_davidson, "seconds")

# Begin Numpy diagonalization of A

start_numpy = time.time()

E,Vec = np.linalg.eig(A)
E = np.sort(E)

end_numpy = time.time()

# End of Numpy diagonalization. Print results.

print("numpy = ", E[:eig],";",
     end_numpy - start_numpy, "seconds") 