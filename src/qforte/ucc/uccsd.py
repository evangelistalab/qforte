import qforte
from qforte.utils import transforms
from qforte.ucc import ucc_helpers
from qforte import vqe


def uccsd_ph_energy_opt(mol, ref, guess_source='ccsd', include_zero_amps=False, maxiter=100):

    num_qubits = len(ref)
    nocc = int(sum(ref))
    nvir = num_qubits - nocc

    if(guess_source == 'ccsd'):
        singles, doubles = mol.get_ccsd_amps()
        T_sq = ucc_helpers.get_uccsd_from_ccsd(nocc, nvir, singles, doubles, make_anti_herm=False, include_zero_amps = include_zero_amps)

    if(guess_source == 'zeros'):
        T_sq = ucc_helpers.get_ucc_zeros_lists(nocc, nvir, order = 2, make_anti_herm=False)

    # print('\nSecond quantized form of T operator, indicates the actual number of parameters needed')
    # for term in T_sq:
    #     print(term)

    myVQE = vqe.UCCVQE(ref, T_sq, mol.get_hamiltonian(), 100 )
    myVQE.do_vqe(maxiter=maxiter)
    Energy = myVQE.get_energy()
    initial_Energy = myVQE.get_inital_guess_energy()

    return Energy, initial_Energy

def singlet_uccsd_ph_energy_opt(mol, ref, guess_source='ccsd', include_zero_amps=False, maxiter=100):

    num_qubits = len(ref)
    nocc = int(sum(ref))
    nvir = num_qubits - nocc

    if(guess_source == 'ccsd'):
        singles, doubles = mol.get_ccsd_amps()
        T_sq = ucc_helpers.get_singlet_uccsd_from_ccsd(nocc, nvir, singles, doubles, make_anti_herm=False, include_zero_amps = include_zero_amps)

    if(guess_source == 'zeros'):
        T_sq = ucc_helpers.get_singlet_ucc_zeros_lists(nocc, nvir, order = 2, make_anti_herm=False)

    # print('\nSecond quantized form of T operator, indicates the actual number of parameters needed')
    # for term in T_sq:
    #     print(term)

    myVQE = vqe.UCCVQE(ref, T_sq, mol.get_hamiltonian(), 100 )
    myVQE.do_vqe(maxiter=maxiter)
    Energy = myVQE.get_energy()
    initial_Energy = myVQE.get_inital_guess_energy()

    return Energy, initial_Energy
