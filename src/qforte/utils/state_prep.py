import qforte

def build_Uprep(ref, trial_state_type):
    Uprep = qforte.QuantumCircuit()
    if trial_state_type == 'reference':
        for j in range(len(ref)):
            if ref[j] == 1:
                Uprep.add_gate(qforte.make_gate('X', j, j))
    else:
        raise ValueError("Only 'reference' supported as state preparation type")


def ref_string(ref, nqb):
    temp = ref.copy()
    temp.reverse()
    ref_basis_idx = int("".join(str(x) for x in temp), 2)
    ref_basis = qforte.QuantumBasis(ref_basis_idx)
    return ref_basis.str(nqb)
