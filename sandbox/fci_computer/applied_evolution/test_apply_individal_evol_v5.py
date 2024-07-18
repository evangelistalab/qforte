# NOTE(Nick): Manually implement the H2 operator...

def t_diff(Tqf, npt, name, print_both=False):
    print(f"\n  ===> {name} Tensor diff <=== ")
    Tnp = qf.Tensor(shape=np.shape(npt), name='Tnp')
    Tnp.fill_from_nparray(npt.ravel(), np.shape(npt))
    if(print_both):
        print(Tqf)
        print(Tnp)
    Tnp.subtract(Tqf)
    print(f"  ||dT||: {Tnp.norm()}")
    if(Tnp.norm() > 1.0e-12):
        print(Tnp)

import qforte as qf
import numpy as np

nel = 4
sz = 0
norb = 4

print(f"nel {nel} norb {norb}")

fc1 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fc2 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fc3 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)




rand = True

if(rand):
    np.random.seed(11)
    random_array = np.random.rand(fc1.get_state().shape()[0], fc1.get_state().shape()[1])
    random = np.array(random_array, dtype = np.dtype(np.complex128))
    # random = np.ones((fc1.get_state().shape()[0], fc1.get_state().shape()[1]))
    Crand = qf.Tensor(fc1.get_state().shape(), "Crand")
    Crand.fill_from_nparray(random.ravel(), Crand.shape())
    rand_nrm = Crand.norm()
    Crand.scale(1/rand_nrm)
    fc1.set_state(Crand)
    fc2.set_state(Crand)
    fc3.set_state(Crand)
else:
    fc1.hartree_fock()
    fc2.hartree_fock()
    fc3.hartree_fock()

# sq_terms = [
#     (+0.704645 * 1.0j, [7, 6], [3, 2]), # 2body ab 
#     (+0.4 * 1.0j, [6], [0]), # 1bdy-a
#     (+0.4 * 1.0j, [7], [3]), # 1bdy-a
#     (+0.704645 * 1.0, [4, 5], [5, 4]), # 1body-nbr ab (coeff must be REAL)
#     (+0.704645 * 1.0, [6, 3], [3, 2]), # 2body-nbr ab 
#     (+0.704645 * 1.0, [2], [2]), # 1body-nbr ab (coeff must be REAL)
#     ]

# Identity
op0 = qf.SQOperator()
op0.add_term(+0.5, [], [])
op0.add_term(+0.5, [], [])
# op1.simplify()
print(op0)

# Identity
op0_v2 = qf.SQOperator()
op0_v2.add_term(+0.5, [], [])
op0_v2.add_term(+0.5, [], [])
op0_v2.simplify()
print(op0_v2)

# ab-double
# -0.704645j [3^ 2^ 7 6] +
# 0.704645j [7^ 6^ 3 2]
op1 = qf.SQOperator()
op1.add_term(-0.704645j, [3, 2], [7, 6])
op1.add_term(0.704645j, [7, 6], [3, 2])
op1.simplify()
print(op1)

# a-ex
# -0.4j [0^ 6] +
# 0.4j [6^ 0]
op2 = qf.SQOperator()
op2.add_term(-0.4j, [0], [6])
op2.add_term(+0.4j, [6], [0])
op2.simplify()
print(op2)

# b-ex
# -0.4j [3^ 7] +
# 0.4j [7^ 3]
op3 = qf.SQOperator()
op3.add_term(-0.4j, [3], [7])
op3.add_term(+0.4j, [7], [3])
op3.simplify()
print(op3)

# a-num-b-num
# -1.40929 [5^ 4^ 5 4]
op4 = qf.SQOperator()
op4.add_term(-0.704645, [5, 4], [5, 4])
op4.add_term(-0.704645, [5, 4], [5, 4])
# op4.simplify()
print(op4)

# NUMBER OP!
# -1.40929 [5^ 4^ 5 4]
op4_v2 = qf.SQOperator()
op4_v2.add_term(-0.704645, [5, 4], [5, 4])
op4_v2.add_term(-0.704645, [5, 4], [5, 4])
op4_v2.simplify()
print(op4_v2)


# 0.704645 [3^ 2^ 6 3] +
# 0.704645 [6^ 3^ 3 2]
op5 = qf.SQOperator()
op5.add_term(+0.704645, [3, 2], [6, 3])
op5.add_term(+0.704645, [6, 3], [3, 2])
op5.simplify()
print(op5)


# 1.40929 [2^ 2]
op6 = qf.SQOperator()
op6.add_term(0.704645, [2], [2])
op6.add_term(0.704645, [2], [2])
# op6.simplify()
print(op6)

# 1.40929 [2^ 2]
op6_v2 = qf.SQOperator()
op6_v2.add_term(0.704645, [2], [2])
op6_v2.add_term(0.704645, [2], [2])
op6_v2.simplify()
print(op6_v2)


# 0.704645 [7^ 5^ 3 1] +
# 0.704645 [1^ 3^ 5 7]
op7 = qf.SQOperator()
op7.add_term(+0.704645, [7, 5], [3, 1])
op7.add_term(+0.704645, [1, 3], [5, 7])
op7.simplify()
print(op7)

# -0.704645 [6^ 4^ 2 0] +
# -0.704645 [0^ 2^ 4 6]
op8 = qf.SQOperator()
op8.add_term(-0.704645, [6, 4], [2, 0])
op8.add_term(-0.704645, [0, 2], [4, 6])
op8.simplify()
print(op8)


# 0.5 [3^ 3]
op9 = qf.SQOperator()
op9.add_term(0.5, [3], [3])
op9.add_term(0.5, [3], [3])
# op9.simplify()
print(op9)

# 0.5 [3^ 3]
op9_v2 = qf.SQOperator()
op9_v2.add_term(0.5, [3], [3])
op9_v2.add_term(0.5, [3], [3])
op9_v2.simplify()
print(op9_v2)


# 0.704645 [4^ 3^ 7 4] +
# 0.704645 [4^ 7^ 3 4]
op10 = qf.SQOperator()
op10.add_term(+0.704645, [4, 3], [7, 4])
op10.add_term(+0.704645, [4, 7], [3, 4])
op10.simplify()
print(op10)



sqops = [
    op0,
    op1,
    op2,
    op3,
    op4,
    op5,
    op6,
    op7,
    op8,
    op9,
]

sqops_v2 = [
    op0_v2,
    op1,
    op2,
    op3,
    op4_v2,
    op5,
    op6_v2,
    op7,
    op8,
    op9_v2,
]

O = qf.SQOperator()
for op in sqops:
    O.add_op(op)

O_v2 = qf.SQOperator()
for op in sqops_v2:
    O_v2.add_op(op)

# print(O)
# print(O_v2)

time = 1.0
print_imag = True

dt = 0.1
N = 10
r = 1
order = 1

hermitian_pairs = qf.SQOpPool()
hermitian_pairs.add_hermitian_pairs(1.0, O)

hermitian_pairs_v2 = qf.SQOpPool()
hermitian_pairs_v2.add_hermitian_pairs(1.0, O_v2)

# gphase = np.exp(-1.0j*dt*mol.nuclear_repulsion_energy)

for i, op in enumerate(sqops):
    print(f"SQOP from O:\n")
    print(op)
    print(f"SQOP from hermitian pair:\n")
    print(hermitian_pairs_v2.terms()[i][1])


for i in range(N):

    fc1.evolve_op_taylor(
        O,
        dt,
        1.0e-15,
        30)

    fc2.evolve_pool_trotter(
        hermitian_pairs_v2,
        dt,
        r,
        order,
        antiherm=False,
        adjoint=False)
    
    # THING THAT MATCHES FQE!!
    for op in sqops:
        fc3.apply_sqop_evolution(dt, op)
    
    # fc2.scale(gphase)

    # print(fc1.get_state().norm())
    # print(fc2.get_state().norm())
    # print(fc3.get_state().norm())

    E1 = np.real(fc1.get_exp_val(O))
    E2 = np.real(fc2.get_exp_val(O))
    E3 = np.real(fc3.get_exp_val(O))

    C1 = fc1.get_state_deep()

    dC2 = fc2.get_state_deep()

    dC3 = fc3.get_state_deep()

    dC2.subtract(C1)
    dC3.subtract(C1)
    
    print(f"t {(i+1)*dt:6.6f} |d-hps| {dC2.norm():6.10f} |d-O| {dC3.norm():6.10f}  {E1:6.6f} {E2:6.6f} {E3:6.6f}")

# print(fc1.str(print_complex=True))
# print(fc2.str(print_complex=True))
# print(fc3.str(print_complex=True))


# SQOP from O:

#  +0.704645 ( 3^ 1^ 7 5 )
#  +0.704645 ( 7^ 5^ 3 1 )

# SQOP from hermitian pair:

#  +0.704645 ( 1^ 3^ 5 7 )
#  +0.704645 ( 5^ 7^ 1 3 )

# SQOP from O:

#  -0.704645 ( 2^ 0^ 6 4 )
#  -0.704645 ( 6^ 4^ 2 0 )

# SQOP from hermitian pair:

#  -0.704645 ( 0^ 2^ 4 6 )
#  -0.704645 ( 4^ 6^ 0 2 )

# t 0.100000 |d-hps| 0.0096659709 |d-O| 0.0096659709  2.409290 2.415935 2.415935
# t 0.200000 |d-hps| 0.0190926162 |d-O| 0.0190813804  2.409290 2.421167 2.421099
# t 0.300000 |d-hps| 0.0281182874 |d-O| 0.0280153019  2.409290 2.424400 2.423807
# t 0.400000 |d-hps| 0.0367348505 |d-O| 0.0362598933  2.409290 2.425741 2.423485
# t 0.500000 |d-hps| 0.0451532305 |d-O| 0.0436400867  2.409290 2.425925 2.420061
# t 0.600000 |d-hps| 0.0538256899 |d-O| 0.0500217061  2.409290 2.426122 2.413974
# t 0.700000 |d-hps| 0.0633952002 |d-O| 0.0553176153  2.409290 2.427677 2.406111
# t 0.800000 |d-hps| 0.0745624094 |d-O| 0.0594915657  2.409290 2.431811 2.397655
# t 0.900000 |d-hps| 0.0879123655 |d-O| 0.0625594711  2.409290 2.439360 2.389887
# t 1.000000 |d-hps| 0.1037811535 |d-O| 0.0645878900  2.409290 2.450595 2.383980

