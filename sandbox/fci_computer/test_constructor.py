import qforte as qf

nel = 4
sz = 0
norb = 4

fci_comp = qf.FCIComputer(nel=nel, sz=sz, norb=norb)

print(fci_comp.str(print_data=True))