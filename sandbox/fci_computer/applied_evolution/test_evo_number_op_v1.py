import qforte as qf
import numpy as np

nel = 4
sz = 0
norb = 4

fci_comp = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fci_comp.hartree_fock()


rand = False
if(rand):
    # random_array = np.random.rand(fci_comp.get_state().shape()[0], fci_comp.get_state().shape()[1])
    # random = np.array(random_array, dtype = np.dtype(np.complex128))
    random = np.ones((fci_comp.get_state().shape()[0], fci_comp.get_state().shape()[1]))
    Crand = qf.Tensor(fci_comp.get_state().shape(), "Crand")
    Crand.fill_from_nparray(random.ravel(), Crand.shape())
    rand_nrm = Crand.norm()
    Crand.scale(1/rand_nrm)
    fci_comp.set_state(Crand)
    print(fci_comp.str(print_data=True))

dim = 2*norb
max_nbody = 1

print("\n Initial FCIcomp Stuff")
print("===========================")
print(fci_comp)


sq_terms = [
    (+0.704645 * 1.0j, [7, 6], [3, 2]), # 2body ab 
    # (+0.4 * 1.0j, [6], [0]), # 1bdy-a
    # (+0.4 * 1.0j, [7], [3]), # 1bdy-a
    # (+0.704645 * 1.0, [6, 3], [3, 2]), # 2body-nbr ab 
    # (+0.704645 * 1.0, [6, 5], [5, 2]), # 2body-nbr ab (coeff must be REAL)
    (+0.704645 * 1.0, [2], [2]), # 1body-nbr ab (coeff must be REAL)
    ]

time = 1.0
print_imag = True

for sq_term in sq_terms:

    sqop = qf.SQOperator()
    sqop.add_term(sq_term[0], sq_term[1], sq_term[2])
    sqop.add_term(np.conj(sq_term[0]), sq_term[2][::-1], sq_term[1][::-1])

    print("\n SQOP Stuff")
    print("===========================")
    print(sqop)
    fci_comp.apply_sqop_evolution(time, sqop)

    print("\n Final FCIcomp Stuff")
    print("===========================")

    Ctemp = fci_comp.get_state_deep()
    cnrm = Ctemp.norm()
    print(f"||C||: {cnrm}")
    print(fci_comp.str(print_data=True, print_complex=print_imag))

# From FQE

# -0.704645j [2^ 3^ 6 7] +
# 0.704645j [7^ 6^ 3 2]
# -0.4j [0^ 6] +
# 0.4j [6^ 0]
# -0.4j [3^ 7] +
# 0.4j [7^ 3]
# 1.40929 [4^ 5^ 5 4]
# 0.704645 [2^ 3^ 3 6] +
# 0.704645 [6^ 3^ 3 2]
# 1.40929 [2^ 2]



# Sector N = 4 : S_z = 0
# a'0011'b'0011' (-0.08209423709643102-0.5122540109764099j)
# a'0011'b'1001' (0.043940936041179816-0.26969973734832664j)
# a'1001'b'0011' (0.19217526069101498-0.41865651763078554j)
# a'1001'b'1001' (-0.5966296163779925+0j)
# a'1010'b'0011' (-0.043940936041179816+0.26969973734832664j)
# a'1010'b'1001' (-0.01857792978321812+0.11402722004633586j)



# [[-0.08209424-0.51225401j  0.        +0.j          0.04394094-0.26969974j
#    0.        +0.j          0.        +0.j          0.        +0.j        ]
#  [ 0.        +0.j          0.        +0.j          0.        +0.j
#    0.        +0.j          0.        +0.j          0.        +0.j        ]
#  [ 0.19217526-0.41865652j  0.        +0.j         -0.59662962+0.j
#    0.        +0.j          0.        +0.j          0.        +0.j        ]
#  [ 0.        +0.j          0.        +0.j          0.        +0.j
#    0.        +0.j          0.        +0.j          0.        +0.j        ]
#  [-0.04394094+0.26969974j  0.        +0.j         -0.01857793+0.11402722j
#    0.        +0.j          0.        +0.j          0.        +0.j        ]
#  [ 0.        +0.j          0.        +0.j          0.        +0.j
#    0.        +0.j          0.        +0.j          0.        +0.j        ]]




