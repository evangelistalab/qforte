import qforte
from qforte.utils import transforms
from qforte.ucc import ucc_helpers
from qforte import vqe


def uccsd_ph_energy_opt(mol, ref, guess_source='ccsd', include_zero_amps=False, maxiter=100):

    # Get params (nqubits, etc.. )
    # norb = mol.nqubits
    num_qubits = len(ref)
    # Energy = 0


    # Get sq_excitations list
    if(guess_source == 'ccsd'):
        singles, doubles = mol.get_ccsd_amps()
        T_sq = ucc_helpers.get_uccsd_from_ccsd(num_qubits, singles, doubles, make_anit_herm=False, include_zero_amps = include_zero_amps)
        print('\nSecond quantized form of T operator, indicates the actual number of parameters needed')
        print('T_sq: ', T_sq)

        #next phases will be done insile VQE and will take T_sq as in input

    # Do VQE
    myVQE = vqe.UCCVQE(ref, T_sq, mol.get_hamiltonian(), 100 )
    myVQE.do_vqe(maxiter=maxiter)
    Energy = myVQE.get_energy()

    # print(help(res))

    return Energy
