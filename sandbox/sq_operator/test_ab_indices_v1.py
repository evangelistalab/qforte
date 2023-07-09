import qforte as qf
# import fqe

# FQE functions
# ops = FermionOperator('3^ 2^ 0 1', 6.9)
# ops += hermitian_conjugated(ops)
# print(f"My operator:\n {ops} \n")
# # largest alfa and beta orbital indicies
# ablk, bblk = fqe.fqe_decorators.largest_operator_index(ops)

print("Compare Operators")
fqe_op_out = " 6.9 [1^ 0^ 2 3] + 6.9 [3^ 2^ 0 1] "
print(f"fqe_op_out {fqe_op_out}")
sqo = qf.SQOperator()
sqo.add_term(6.9, [3, 2], [0, 1])
sqo.add_term(6.9, [1, 0], [2, 3])
print(sqo)
print("")


print("Compare largers ab indices")
fqe_ablk_bblk_out = "2, 3"
print(f"fqe_ablk_bblk_out {fqe_ablk_bblk_out}")

ablk, bblk = sqo.get_largest_alfa_beta_indices()
print(f"ablk: {ablk}")
print(f"bblk: {bblk}")
print("")