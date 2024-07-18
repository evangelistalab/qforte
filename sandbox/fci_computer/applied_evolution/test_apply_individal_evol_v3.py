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

# print("\n Initial FCIcomp Stuff")
# print("===========================")
# print(fci_comp)

loaded_data = np.load('wfn_evo_individ_ops_v1.npz')
fqe_wfn1 = loaded_data['frm_hf_op1']
fqe_wfn2 = loaded_data['frm_hf_op2']
fqe_wfn3 = loaded_data['frm_hf_op3']
fqe_wfn4 = loaded_data['frm_hf_op4']
fqe_wfn5 = loaded_data['frm_hf_op5']
fqe_wfn6 = loaded_data['frm_hf_op6']
fqe_wfn_all = loaded_data['frm_hf_all']

fqe_individ_wfns = [
    fqe_wfn1,
    fqe_wfn2,
    fqe_wfn3,
    fqe_wfn4,
    fqe_wfn5,
    fqe_wfn6,
]


# sq_terms = [
#     (+0.704645 * 1.0j, [7, 6], [3, 2]), # 2body ab 
#     (+0.4 * 1.0j, [6], [0]), # 1bdy-a
#     (+0.4 * 1.0j, [7], [3]), # 1bdy-a
#     (+0.704645 * 1.0, [4, 5], [5, 4]), # 1body-nbr ab (coeff must be REAL)
#     (+0.704645 * 1.0, [6, 3], [3, 2]), # 2body-nbr ab 
#     (+0.704645 * 1.0, [2], [2]), # 1body-nbr ab (coeff must be REAL)
#     ]

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
# op6.add_term(1.40929, [2], [2])
# op6.add_term(1.40929, [2], [2])
op6.add_term(0.704645, [2], [2])
op6.add_term(0.704645, [2], [2])
# op6.simplify()
print(op6)

sqops = [
    op1,
    op2,
    op3,
    op4,
    op5,
    op6,
]



time = 1.0
print_imag = True

for i, sqop in enumerate(sqops):

    fci_comp.hartree_fock()

    # print("\n SQOP Stuff")
    # print("===========================")
    # print(sqop)
    fci_comp.apply_sqop_evolution(time, sqop)

    # print("\n Final FCIcomp Stuff")
    # print("===========================")

    Ctemp = fci_comp.get_state_deep()
    cnrm = Ctemp.norm()
    # print(f"||C||: {cnrm}")
    # print(fci_comp.str(print_data=True, print_complex=print_imag))

    t_diff(
        fci_comp.get_state_deep(), 
        fqe_individ_wfns[i], 
        f"|| Cqc_{i+1} - Cfqe_{i+1} ||", 
        print_both=False)


# now try all of them...
fci_comp.hartree_fock()   
for i, sqop in enumerate(sqops):

    # print("\n SQOP Stuff")
    # print("===========================")
    # print(sqop)
    fci_comp.apply_sqop_evolution(time, sqop)

    # print("\n Final FCIcomp Stuff")
    # print("===========================")

    Ctemp = fci_comp.get_state_deep()
    cnrm = Ctemp.norm()
    # print(f"||C||: {cnrm}")
    # print(fci_comp.str(print_data=True, print_complex=print_imag))

t_diff(
    fci_comp.get_state_deep(), 
    fqe_wfn_all, 
    f"|| Cqc_all - Cfqe_all ||", 
    print_both=False)


# print(fci_comp.str(print_data=True, print_complex=print_imag))






