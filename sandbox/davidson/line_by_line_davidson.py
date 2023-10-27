import numpy as np

# n = 1000 # was 1200				# Dimension of CI matrix (CI vector will be this size)

# ''' Create sparse, diagonally dominant matrix A with 
# 	diagonal containing 1,2,3,...n. The eigenvalues
#     should be very close to these values. You can 
#     change the sparsity. A smaller number for sparsity
#     increases the diagonal dominance. Larger values
#     (e.g. sparsity = 1) create a dense matrix
# '''

n = 5

sparsity = 0.0001
A = np.zeros((n,n)) # creates an empty square matrix of dimension n

for i in range(0,n): # for loop adds numbers to diagonal of square matrix corresponding to index, therefore diagonal is filled with increasing integers
    A[i,i] = i + 1 
A = A + sparsity*np.random.randn(n,n) # creates another square matrix of dimension n contaning random small numbers and adds it to original matrix
A = (A.T + A)/2 # adds matrix A to the transpose of itself and divides the resultant matrix by 2. 

print(A)



t = np.eye(n,5) # set of k unit vectors as guess  

print(t)

print(t[:,0])

print(t[:,0]/np.linalg.norm(t[:,0]))



# A = np.random.randn(4,4)
# print(f'{A}\n')

# B = sparsity * A
# print(f'{B}\n')

# A = A + B 
# print(f'{A}\n')

# print(f'{A.T}\n')



