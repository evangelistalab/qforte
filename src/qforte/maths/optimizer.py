import qforte

import copy
import numpy as np

def diis_solve(self, residual):
    """This function attempts to minimizes the norm of the residual vector
    by using a quasi-Newton uptate procedure for the amplutudes paired with
    the direct inversion of iterative subspace (DIIS) convergece acceleration.
    """
    # draws heavy inspiration from Daniel Smith's ccsd_diss.py code in psi4 numpy
    diis_dim = 0
    t_diis = [copy.deepcopy(self._tamps)]
    e_diis = []
    rk_norm = 1.0
    Ek0 = self.energy_feval(self._tamps)

    print('\n    k iteration         Energy               dE           Nrvec ev      Nrm ev*        ||r||')
    print('---------------------------------------------------------------------------------------------------')

    if (self._print_summary_file):
        f = open("summary.dat", "w+", buffering=1)
        f.write('\n#    k iteration         Energy               dE           Nrvec ev      Nrm ev*        ||r||')
        f.write('\n#--------------------------------------------------------------------------------------------------')
        f.close()

    for k in range(1, self._opt_maxiter+1):

        t_old = copy.deepcopy(self._tamps)

        #do regular update
        r_k = residual(self._tamps)
        rk_norm = np.linalg.norm(r_k)

        r_k = self.get_res_over_mpdenom(r_k)
        self._tamps = list(np.add(self._tamps, r_k))

        Ek = self.energy_feval(self._tamps)
        dE = Ek - Ek0
        Ek0 = Ek

        self._res_vec_evals += 1
        self._res_m_evals += len(self._tamps)

        print(f'     {k:7}        {Ek:+12.10f}      {dE:+12.10f}      {self._res_vec_evals:4}        {self._res_m_evals:6}       {rk_norm:+12.10f}')

        if (self._print_summary_file):
            f = open("summary.dat", "a", buffering=1)
            f.write(f'\n     {k:7}        {Ek:+12.12f}      {dE:+12.12f}      {self._res_vec_evals:4}        {self._res_m_evals:6}       {rk_norm:+12.12f}')
            f.close()

        if(rk_norm < self._opt_thresh):
            self._Egs = Ek
            break

        t_diis.append(copy.deepcopy(self._tamps))
        e_diis.append(np.subtract(copy.deepcopy(self._tamps), t_old))

        if(k >= 1):
            diis_dim = len(t_diis) - 1

            # Construct diis B matrix (following Crawford Group github tutorial)
            B = np.ones((diis_dim+1, diis_dim+1)) * -1
            bsol = np.zeros(diis_dim+1)

            B[-1, -1] = 0.0
            bsol[-1] = -1.0
            for i, ei in enumerate(e_diis):
                for j, ej in enumerate(e_diis):
                    B[i,j] = np.dot(np.real(ei), np.real(ej))

            B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()

            x = np.linalg.solve(B, bsol)

            t_new = np.zeros(( len(self._tamps) ))
            for l in range(diis_dim):
                temp_ary = x[l] * np.asarray(t_diis[l+1])
                t_new = np.add(t_new, temp_ary)

            self._tamps = copy.deepcopy(t_new)

    self._n_classical_params = self._n_classical_params = len(self._tamps)
    self._n_cnot = self.build_Uvqc().get_num_cnots()
    self._n_pauli_trm_measures += 2*self._Nl*k*len(self._tamps) + self._Nl*k
    self._Egs = Ek

