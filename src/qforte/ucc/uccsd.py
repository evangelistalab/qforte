import qforte
from qforte.utils import transforms
from qforte.ucc import ucc_helpers
from qforte import vqe


def ucc_energy(mol, ref, order, fast=False, guess_source='zeros', include_zero_amps=False, maxiter=100):

    if(mol.multiplicity != 1):
        raise ValueError('UCC calculatinos for higher than singlet multiplicity not currently supported.')

    num_qubits = len(ref)
    nocc = int(sum(ref))
    nvir = num_qubits - nocc

    if(guess_source == 'ccsd'):
        singles, doubles = mol.get_ccsd_amps()
        T_sq = ucc_helpers.get_ucc_from_ccsd(nocc, nvir, singles, doubles, order = order, make_anti_herm=False, include_zero_amps = include_zero_amps)

    elif(guess_source == 'zeros'):
        T_sq = ucc_helpers.get_ucc_zeros_lists(nocc, nvir, order = order, make_anti_herm=False)

    else:
        raise ValueError("Must specify amplitude guess source: accepted values are 'zeros' and 'ccsd'. ")

    print('\nSecond quantized form of T operator, indicates the actual number of parameters needed')
    for term in T_sq:
        print(term)

    myVQE = vqe.UCCVQE(ref, T_sq, mol.get_hamiltonian(), 100 )
    myVQE.do_vqe(maxiter=maxiter, fast=fast)
    Energy = myVQE.get_energy()
    initial_Energy = myVQE.get_inital_guess_energy()

    return Energy, initial_Energy

def uccsd_energy(mol, ref, fast=False, guess_source='ccsd', include_zero_amps=False, maxiter=100):

    Energy, initial_Energy = ucc_energy(mol, ref, 2, fast=fast, guess_source=guess_source,
                                        include_zero_amps=include_zero_amps, maxiter=100)

    return Energy, initial_Energy
