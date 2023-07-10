import qforte as qf

dim = 4
max_nbody = 1
is_spatial = False
is_restricted = False

Top = qf.TensorOperator(
    max_nbody = max_nbody, 
    dim = dim
    )

sqop = qf.SQOperator()
sqop.add_term(1.0, [2], [0])
sqop.add_term(1.0, [0], [2])

for term2 in sqop.terms():
    print(term2)

print(f"sqop.ranks_present(): {sqop.ranks_present()}")

ablk, bblk = sqop.get_largest_alfa_beta_indices()
print(f"ablk: {ablk}")
print(f"bblk: {bblk}")

print(sqop)
# print(Top)

# first should print tensor dim
print("Now FQE reference intermediates")
print("========================")
tensor_dim_ref = "[4, 4]"
print(f"tensor_dim_ref: \n{tensor_dim_ref}")

index_mask_ref = "[0, 0]"
print(f"index_mask_ref: \n{index_mask_ref}")

index_dict_dagger_ref = "[[0, 0]]" 
print(f"index_dict_dagger_ref: \n{index_dict_dagger_ref}")

index_dict_nondagger_ref = "[[0, 1]]" 
print(f"index_dict_nondagger_ref: \n{index_dict_nondagger_ref}")

index_mask_term1_ref = "[1, 0]" 
print(f"index_mask_term1_ref: \n{index_mask_term1_ref}")

index_mask_term2_ref = "[0, 1]" 
print(f"index_mask_term2_ref: \n{index_mask_term2_ref}")



print("\n\n")

print("Now printing qf Debug Stuff")
print("===========================")
Top.add_sqop_of_rank(sqop, 2)

print(Top)