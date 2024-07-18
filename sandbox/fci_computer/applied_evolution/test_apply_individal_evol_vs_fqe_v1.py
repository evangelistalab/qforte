# NOTE(Nick): A more rigerous comparision for evolving individual SQOperators for the FCI Computer,
# Special attention is paid to the difference between evolving under:
#  (i) operators with no number operator contrubtions, 
#  (ii) operators that are number operators
#  (iii) operators that contain a numer operator and a non number operator 

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

# -0.704645j [3^ 2^ 7 6] +
# 0.704645j [7^ 6^ 3 2]
op1 = qf.SQOperator()
op1.add_term(-0.704645j, [3, 2], [7, 6])
op1.add_term(+0.704645j, [7, 6], [3, 2])
op1.simplify()
print(op1)

# -0.4j [0^ 6] +
# 0.4j [6^ 0]
op2 = qf.SQOperator()
op2.add_term(-0.4j, [0], [6])
op2.add_term(+0.4j, [6], [0])
op2.simplify()
print(op2)

# -0.4j [3^ 7] +
# 0.4j [7^ 3]
op3 = qf.SQOperator()
op3.add_term(-0.4j, [3], [7])
op3.add_term(+0.4j, [7], [3])
op3.simplify()
print(op3)

# NUMBER OP!
# -1.40929 [5^ 4^ 5 4]
op4 = qf.SQOperator()
op4.add_term(-0.704645, [5, 4], [5, 4])
op4.add_term(-0.704645, [5, 4], [5, 4])
# op4.simplify()
print(op4)


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

sqops = [
    op0,
    op1,
    op2,
    op3,
    op4,
    op5,
    op6,
]

sqops_v2 = [
    op0_v2,
    op1,
    op2,
    op3,
    op4,
    op5,
    op6_v2,
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

    # print(fc1.str(print_complex=True))
    # print(fc2.str(print_complex=True))

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
    
    print(f"t {(i+1)*dt:6.6f} |d-hps| {dC2.norm():6.6f} |d-O| {dC3.norm():6.6f}  {E1:6.6f} {E2:6.6f} {E3:6.6f}")






