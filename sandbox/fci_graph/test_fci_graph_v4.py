import qforte as qf
import numpy as np

na = 3
nb = 3
norb = 6


qg = qf.FCIGraph(na, nb, norb)

daga = [2, 3]
undaga = [4, 5]

dagb = [1]
undagb = [3]

[vala, sourcea, targeta, paritya] = qg.make_mapping_each(
    True,
    daga,
    undaga,
)

[valb, sourceb, targetb, parityb] = qg.make_mapping_each(
    False,
    dagb,
    undagb,
)

print(f"acount {vala}")
print(f" sourcea: {sourcea}")
print(f" targeta: {targeta}")
print(f" paritya: {paritya}")

print(f"acount {valb}")
print(f" sourceb: {sourceb}")
print(f" targetb: {targetb}")
print(f" parityb: {parityb}")
#LGTM! (Nick)

# Correct Output
"""
vala 2
valb 6

 ==> result alfa <== 
 [[ 9. 13.  1.]
 [15. 14.  1.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]]

 ==> result beta <== 
 [[ 4.  7.  1.]
 [ 7. 19.  0.]
 [ 8. 35.  0.]
 [16. 22.  1.]
 [17. 38.  1.]
 [19. 50.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]]
"""

