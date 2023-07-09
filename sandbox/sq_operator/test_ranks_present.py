import qforte as qf
# import fqe

# FQE functions
# ops = FermionOperator('3^ 2^ 0 1', 6.9)
# ops += hermitian_conjugated(ops)
# print(f"My operator:\n {ops} \n")
# # largest alfa and beta orbital indicies
# ablk, bblk = fqe.fqe_decorators.largest_operator_index(ops)

print("Compare Operators")
# fqe_op_out = " 6.9 [1^ 0^ 2 3] + 6.9 [3^ 2^ 0 1] "
# print(f"fqe_op_out {fqe_op_out}")
sqo = qf.SQOperator()
sqo.add_term(6.9, [3, 2], [0, 1])
sqo.add_term(6.9, [1, 0], [2, 3])
sqo.add_term(4.2, [], [2])
sqo.add_term(1.2, [], [])
sqo.add_term(7.2, [3], [6])

print(sqo)
print("")

print("Compare ranks present")
should_be = "[4, 1, 0, 2]"
print(f"should_be: {should_be}")

ranks_present = sqo.ranks_present()
print(f"qf_ranks_present: {ranks_present}")
print("")

# lgtm (Nick)