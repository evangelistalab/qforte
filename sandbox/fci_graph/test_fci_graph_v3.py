import qforte as qf
import numpy as np

na = 2
nb = 2
norb = 4


qg = qf.FCIGraph(na, nb, norb)

daga = [0]
undaga = [2]

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





