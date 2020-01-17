import qforte
from qforte import vqe

def adaptvqe_energy(ref, mol, avqe_thresh,
                    print_pool=True,
                    trotter_number=1,
                    fast=True,
                    optimizer='BFGS',
                    opt_thresh=1.0e-5,
                    use_analytic_grad=True,
                    adapt_mxitr=50,
                    opt_mxitr=200):

    print('\n-----------------------------------------------------')
    print('  Adaptive Derivative-Assembled Pseudo-Trotter VQE   ')
    print('-----------------------------------------------------')

    nqubits = len(ref)
    init_basis_idx = qforte.qkd.qk_helpers.ref_to_basis_idx(ref)
    init_basis = qforte.QuantumBasis(init_basis_idx)

    print('\n\n                 ==> ADAPT-VQE options <==')
    print('-----------------------------------------------------------')
    print('Reference state:                         ',  init_basis.str(nqubits))
    print('ADAPT-VQE grad-norm threshold (eps):     ',  avqe_thresh)
    print('Optimizer grad-norm threshold (theta):   ',  opt_thresh)
    print('Trotter number (m):                      ',  trotter_number)
    print('Use fast version of algorithm:           ',  str(fast))
    print('Type of optimization:                    ',  optimizer)
    print('Use analytic gradient:                   ',  str(use_analytic_grad))
    #TODO: enable finite measurement per term (Nick)
    print('Number of measurements per term:         ',  'infinite')

    N_meas_per_op = 100
    myAVQE = vqe.ADAPTVQE(ref, mol.get_hamiltonian(), avqe_thresh,
                      optimizer = optimizer,
                      opt_thresh = opt_thresh,
                      N_samples=N_meas_per_op,
                      use_analytic_grad=use_analytic_grad,
                      use_fast_measurement=fast,
                      trott_num=trotter_number)

    myAVQE.fill_pool()
    if print_pool:
        myAVQE._pool_obj.print_pool()

    myAVQE.fill_comutator_pool()

    avqe_iter = 0
    hit_maxiter = 0
    while not myAVQE._converged:

        print('\n\n -----> ADAPT-VQE iteration ', avqe_iter, ' <-----\n')
        myAVQE.update_ansatz()

        if myAVQE._converged:
            break

        print('\ntoperators included from pool: \n', myAVQE._tops)
        print('tamplitudes for tops: \n', myAVQE._tamps)

        myAVQE.solve(fast=fast, opt_maxiter=opt_mxitr)
        avqe_iter += 1

        if avqe_iter > adapt_mxitr-1:
            hit_maxiter = 1
            break

    if hit_maxiter:
        final_energy = myAVQE.get_final_energy(hit_max_avqe_iter=1)

    final_energy = myAVQE.get_final_energy()

    print('\n\n                  ==> ADAPT-VQE summary <==')
    print('---------------------------------------------------------------')
    print('Final ADAPT-VQE Energy:                     ', round(final_energy, 10))
    print('Number of operators in pool:                 ', len(myAVQE._pool))
    print('Final number of amplitudes in ansatz:        ', len(myAVQE._tamps))
    print('Total number of Hamiltonian measurements:    ', myAVQE.get_num_ham_measurements())
    print('Total number of comutator measurements:      ', myAVQE.get_num_comut_measurements())

    return final_energy
