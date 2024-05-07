"""
Classes for Gibbs State ADAPT-VQE
====================================
"""

import qforte as qf

from qforte.abc.uccvqeabc import UCCVQE

import numpy as np
import scipy
import copy

kb = 3.166811563455546e-06

class Gibbs_ADAPT(UCCVQE):
    def run(self,
            references = None,
            pool_type = "GSD",
            verbose = False,
            T = 273):
        
        self._references = references
        self.T = T
        if self.T != 0:
            self.beta = 1/(kb*T)
        
        self._pool_type = pool_type
        self.verbose = verbose
        self._Upreps = [qf.build_Uprep(ref, 'occupation_list') for ref in references] 
        self.fill_pool() 


    def compute_free_energy(self, x):
        #Get energy expectation values.
        Es = []
        for i, reference in enumerate(self._references):
            self._Uprep = self._Upreps[i]
            self.energy_feval(x)
            Es.append(self.energy_feval(x))
        Es = np.array(Es)

        #Re-sort so that E0 is the lowest.
        idx = np.argsort(Es)
        self._Upreps = [self._Upreps[i] for i in idx]
        Es = Es[idx]
        if self.T == 0:
            return Es[0]

        #Compute energy.
        deltas = np.ones(Es.shape)*Es[0] - Es
        gamma = np.exp(deltas*self.beta)
        tot_gamma = np.sum(gamma)
        omega = gamma/tot_gamma
        F = np.sum(omega*Es)
        #F += np.sum(omega*deltas)
        #F -= np.log(tot_gamma)/self.beta
        return F

    def compute_free_energy_gradient(self, x):
        #Get energy expectation values and gradients.
        Es = []
        E_grads = []
        for i, reference in enumerate(self._references):
            self._Uprep = self._Upreps[i]
            self.energy_feval(x)
            Es.append(self.energy_feval(x))
            E_grads.append(self.measure_gradient(x))

        Es = np.array(Es)
        E_grads = np.array(E_grads)
        

        #Re-sort so that E0 is the lowest.
        idx = np.argsort(Es)
        self._Upreps = [self._Upreps[i] for i in idx]
        Es = Es[idx]
        E_grads = E_grads[idx]
        if self.T == 0:
            return E_grads[0]

        #Compute omegas and their derivatives
        deltas = np.ones(Es.shape)*Es[0] - Es
        
        d_deltas = np.ones(E_grads.shape)*E_grads[0,:] - E_grads
        
        gamma = np.exp(deltas*self.beta)
        tot_gamma = np.sum(gamma)
        omega = gamma/tot_gamma
        d_omega = (self.beta/tot_gamma) * np.einsum('j,j,ju->ju', gamma, deltas, d_deltas)
        d_omega -= (self.beta/(tot_gamma**2)) * np.einsum('j,i,i,iu->ju', gamma, gamma, deltas, d_deltas)

        F = np.sum(omega*Es)
        #F -= np.log(tot_gamma)/self.beta
        return omega, d_omega
        

        dF = np.einsum('j,ju->u', omega, E_grads)
        dF += np.einsum('ju,j->u', d_omega, Es)
        #dF += (1/self.beta)*np.einsum('ju->u', d_omega) 
        #dF += (self.beta/tot_gamma)*np.einsum('j,j,j,ju->u', gamma, deltas, deltas, d_deltas)
        #dF -= (np.log(tot_gamma)/tot_gamma)*np.einsum('j,j,ju->u', gamma, deltas, d_deltas)
        #dF -= (self.beta/tot_gamma**2)*np.einsum('i,iu,j,j->u', deltas, d_deltas, gamma, deltas)
        #dF += (np.log(tot_gamma)/tot_gamma**2)*np.einsum('i,iu,j->u', deltas, d_deltas, gamma)
        

        return dF 
        
    def free_energy_vqe(self, x0):
        #Run a VQE to minimize x.
        self.vqe_iteration = 0
        print(f"0 {self.compute_free_energy(x0)}")
        res = scipy.optimize.minimize(self.compute_free_energy,
                                      x0,
                                      method = "BFGS",
                                      jac = self.compute_free_energy_gradient,
                                      options = {'gtol': 1e-6, 'disp': True},
                                      callback = self.callback
                                      )        
        assert res.success == True
        return res.fun, res.x

    def callback(self, x):
        self.vqe_iteration += 1
        print(f"{self.vqe_iteration} {self.compute_free_energy(x)}")
        

    def get_num_commut_measurements(self):
        pass
    def get_num_ham_measurements(self):
        pass
    def print_options_banner(self):
        pass
    def print_summary_banner(self):
        pass
    def run_realistic(self):
        pass
    def solve(self):
        pass
    def verify_run(self):
        pass    
                   


        



        
