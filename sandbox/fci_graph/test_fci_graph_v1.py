import qforte as qf

na = 2
nb = 2
norb = 6


qg = qf.FCIGraph(na, nb, norb)

print(f"qg.get_nalfa(): {qg.get_nalfa()}")

print(f"qg.get_astr(): {qg.get_astr()}")
print(f"qg.get_bstr(): {qg.get_bstr()}")

print(f"qg.get_aind(): {qg.get_aind()}")

alfa_map = qg.get_alfa_map()

# print(f"alfa map: {alfa_map}")

# for k, v in alfa_map.items():
#     print(f"{k}")
#     for thing in v:
#         print(f" {thing}")
#     print("")

dexca = qg.get_dexca()
# print(f"qg.get_dexca(): {dexca}")

print("")
for vi in dexca:
    print("")
    for vj in vi:
        print(vj)


"""
nel: 4
sz: 0
norb: 4
nalfa choose nalfa_orbs: 6
nbeta choose nbeta_orbs: 6

FCI Graph Info: 


  g._astr: [ 3  5  9  6 10 12]
  g._bstr: [ 3  5  9  6 10 12]
  type(g._bstr): <class 'numpy.ndarray'>

  g._aind: {3: 0, 5: 1, 9: 2, 6: 3, 10: 4, 12: 5}
  g._bind: {3: 0, 5: 1, 9: 2, 6: 3, 10: 4, 12: 5}
  type(g._bind): <class 'dict'>

  g._alfa_map:

   (0, 0) 
 [[0 0 1]
 [1 1 1]
 [2 2 1]] 

   (0, 1) 
 [[3 1 1]
 [4 2 1]] 

   (0, 2) 
 [[ 3  0 -1]
 [ 5  2  1]] 

   (0, 3) 
 [[ 4  0 -1]
 [ 5  1 -1]] 

   (1, 0) 
 [[1 3 1]
 [2 4 1]] 

   (1, 1) 
 [[0 0 1]
 [3 3 1]
 [4 4 1]] 

   (1, 2) 
 [[1 0 1]
 [5 4 1]] 

   (1, 3) 
 [[ 2  0  1]
 [ 5  3 -1]] 

   (2, 0) 
 [[ 0  3 -1]
 [ 2  5  1]] 

   (2, 1) 
 [[0 1 1]
 [4 5 1]] 

   (2, 2) 
 [[1 1 1]
 [3 3 1]
 [5 5 1]] 

   (2, 3) 
 [[2 1 1]
 [4 3 1]] 

   (3, 0) 
 [[ 0  4 -1]
 [ 1  5 -1]] 

   (3, 1) 
 [[ 0  2  1]
 [ 3  5 -1]] 

   (3, 2) 
 [[1 2 1]
 [3 4 1]] 

   (3, 3) 
 [[2 2 1]
 [4 4 1]
 [5 5 1]] 


  g._dexca:
 [[[ 0  0  1]
  [ 3  2 -1]
  [ 4  3 -1]
  [ 0  5  1]
  [ 1  6  1]
  [ 2  7  1]]

 [[ 1  0  1]
  [ 3  1  1]
  [ 5  3 -1]
  [ 0  9  1]
  [ 1 10  1]
  [ 2 11  1]]

 [[ 2  0  1]
  [ 4  1  1]
  [ 5  2  1]
  [ 0 13  1]
  [ 1 14  1]
  [ 2 15  1]]

 [[ 1  4  1]
  [ 3  5  1]
  [ 5  7 -1]
  [ 0  8 -1]
  [ 3 10  1]
  [ 4 11  1]]

 [[ 2  4  1]
  [ 4  5  1]
  [ 5  6  1]
  [ 0 12 -1]
  [ 3 14  1]
  [ 4 15  1]]

 [[ 2  8  1]
  [ 4  9  1]
  [ 5 10  1]
  [ 1 12 -1]
  [ 3 13 -1]
  [ 5 15  1]]]

  LGTM!
"""
