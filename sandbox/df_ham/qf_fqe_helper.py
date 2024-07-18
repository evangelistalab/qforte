import qforte as qf
import numpy as np

def get_fqe_individ_op_strings(sqop):

    # print("\n\n#==> Begin Op String <==\n")

    fqe_str = ""

    for i, term in enumerate(sqop.terms()):

        # fqe_str += f"\n# qf op: {term}\n"

        if (len(term[1]) != len(term[2])) :
            print(f'bad term: {term}')
            raise ValueError("len(term[1]) != len(term[2])")
        
        if(len(term[1])==0):
            fqe_str += f"\ne0 = {np.real(term[0])} \n"

        if(len(term[1])==1):
           
            temp_str_1bdy = f"""
op{i} = of.FermionOperator(
    "{term[1][0]}^ {term[2][0]}",
    {np.real(term[0])})

op{i} = nod(op{i})
            """

            fqe_str += temp_str_1bdy

        if(len(term[1])==2):
           
            temp_str_2bdy = f"""
op{i} = of.FermionOperator(
    "{term[1][0]}^ {term[1][1]}^ {term[2][0]} {term[2][1]}",
    {np.real(term[0])})

op{i} = nod(op{i})
            """

            fqe_str += temp_str_2bdy

    return fqe_str

def get_fqe_op_string(sqop):

    # print("\n\n#==> Begin Op String <==\n")

    fqe_str = "\nop = of.FermionOperator()"

    for i, term in enumerate(sqop.terms()):

        # fqe_str += f"\n# qf op: {term}\n"

        if (len(term[1]) != len(term[2])) :
            print(f'bad term: {term}')
            raise ValueError("len(term[1]) != len(term[2])")
        
        if(len(term[1])==0):
            fqe_str += f"\ne0 = {np.real(term[0])} \n"

        if(len(term[1])==1):
           
            temp_str_1bdy = f"""
op{i} = of.FermionOperator(
    "{term[1][0]}^ {term[2][0]}",
    {np.real(term[0])})

op{i} = nod(op{i})

op += op{i}
            """

            fqe_str += temp_str_1bdy

        if(len(term[1])==2):
           
            temp_str_2bdy = f"""
op{i} = of.FermionOperator(
    "{term[1][0]}^ {term[1][1]}^ {term[2][0]} {term[2][1]}",
    {np.real(term[0])})

op{i} = nod(op{i})

op += op{i}
            """

            fqe_str += temp_str_2bdy

    return fqe_str

def get_fqe_op_string_no_normal_order(sqop):

    # print("\n\n#==> Begin Op String <==\n")

    fqe_str = "\nop = of.FermionOperator()"

    for i, term in enumerate(sqop.terms()):

        # fqe_str += f"\n# qf op: {term}\n"

        if (len(term[1]) != len(term[2])) :
            print(f'bad term: {term}')
            raise ValueError("len(term[1]) != len(term[2])")
        
        if(len(term[1])==0):
            fqe_str += f"\ne0 = {np.real(term[0])} \n"

        if(len(term[1])==1):
           
            temp_str_1bdy = f"""
op{i} = of.FermionOperator(
    "{term[1][0]}^ {term[2][0]}",
    {np.real(term[0])})

op += op{i}
            """

            fqe_str += temp_str_1bdy

        if(len(term[1])==2):
           
            temp_str_2bdy = f"""
op{i} = of.FermionOperator(
    "{term[1][0]}^ {term[1][1]}^ {term[2][0]} {term[2][1]}",
    {np.real(term[0])})

op += op{i}
            """

            fqe_str += temp_str_2bdy

    return fqe_str


def get_fqe_hp_string(hpairs):

    print("\n\n#==> Begin Herm Pair String <==\n")

    fqe_str_2 = "hp_lst = []"

    for l, sqop in enumerate(hpairs.terms()):
        temp = get_fqe_op_string_no_normal_order(sqop[1])
        fqe_str_2 += "\n" + temp
        fqe_str_2 += "\nhp_lst.append(op)"

    return fqe_str_2