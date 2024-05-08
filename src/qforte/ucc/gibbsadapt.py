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
        try:
            assert(self.T > 0)
        except:
            print("T must be positive.")
            exit()

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
        #Compute energy.
        deltas = np.ones(Es.shape)*Es[0] - Es
        gamma = np.exp(deltas*self.beta)
        tot_gamma = np.sum(gamma)
        omega = gamma/tot_gamma
        #F = np.sum(omega*(Es)
        #F += np.sum(omega*deltas)
        F = np.sum(omega * Es[0])
        F -= np.log(tot_gamma)/self.beta
        return F

    def compute_free_energy_gradient(self, x):
        #Get energy expectation values and gradients.
        Es = []
        E_grads = []
        for i, reference in enumerate(self._references):
            self._Uprep = self._Upreps[i]
            Es.append(self.energy_feval(x))
            E_grads.append(self.measure_gradient(x))

        Es = np.array(Es)
        E_grads = np.array(E_grads)
        #Re-sort so that E0 is the lowest.
        idx = np.argsort(Es)
        self._Upreps = [self._Upreps[i] for i in idx]
        Es = Es[idx]
        E_grads = E_grads[idx]
        
        #Compute omegas and their derivatives
        deltas = np.ones(Es.shape)*Es[0] - Es
        
        d_deltas = -E_grads
        for i in range(len(self._Upreps)):
            d_deltas[i,:] += E_grads[0,:]
        
        gamma = np.exp(deltas*self.beta)
        tot_gamma = np.sum(gamma)
        omega = gamma/tot_gamma
        d_gamma = self.beta*np.einsum('i,iu->iu', gamma, d_deltas)
        tot_d_gamma = np.einsum('iu->u', d_gamma)
        
        
        d_omega = (1/tot_gamma)*(d_gamma - np.einsum('i,u->iu', omega, tot_d_gamma))
        
        
        dF = np.einsum('iu,i->u', d_omega, Es + deltas - (np.log(tot_gamma)/self.beta)*np.ones(omega.shape))
        #dF = np.einsum('iu,i->u', d_omega, Es)
        #dF += np.einsum('iu,i->u', d_omega, deltas)

        dF += np.einsum('i,iu->u', omega, E_grads + d_deltas)
        #dF += np.einsum('i,iu->u', omega, E_grads) 
        #dF += np.einsum('i,iu->u', omega, d_deltas)
        #dF -= (np.log(tot_gamma)/self.beta)*np.einsum('iu->u', d_omega)
        dF -= (1/(self.beta*tot_gamma))*np.einsum('i,u->u', omega, tot_d_gamma)
        
        return dF

    def compute_addition_gradient(self):
        Es = [] 
        E_grads = []
        for i, reference in enumerate(self._references):
            self._Uprep = self._Upreps[i]
            Es.append(self.energy_feval(self._tamps))
            E_grads.append(self.measure_gradient3())

        Es = np.array(Es)
        E_grads = np.array(E_grads)
         
        #Re-sort so that E0 is the lowest.
        idx = np.argsort(Es)
        self._Upreps = [self._Upreps[i] for i in idx]
        Es = Es[idx]
        E_grads = E_grads[idx]
        
        #Compute omegas and their derivatives
        deltas = np.ones(Es.shape)*Es[0] - Es
        
        d_deltas = -E_grads
        for i in range(len(self._Upreps)):
            d_deltas[i,:] += E_grads[0,:]
        
        gamma = np.exp(deltas*self.beta)
        
        tot_gamma = np.sum(gamma)
        omega = gamma/tot_gamma
        d_gamma = self.beta*np.einsum('i,iu->iu', gamma, d_deltas)
        tot_d_gamma = np.einsum('iu->u', d_gamma)
        
        d_omega = (1/tot_gamma)*(d_gamma - np.einsum('i,u->iu', omega, tot_d_gamma))
        
        dF = np.einsum('iu,i->u', d_omega, Es + deltas - (np.log(tot_gamma)/self.beta)*np.ones(omega.shape))
        #dF = np.einsum('iu,i->u', d_omega, Es)
        #dF += np.einsum('iu,i->u', d_omega, deltas)

        dF += np.einsum('i,iu->u', omega, E_grads + d_deltas)
        #dF += np.einsum('i,iu->u', omega, E_grads) 
        #dF += np.einsum('i,iu->u', omega, d_deltas)
        #dF -= (np.log(tot_gamma)/self.beta)*np.einsum('iu->u', d_omega)
        dF -= (1/(self.beta*tot_gamma))*np.einsum('i,u->u', omega, tot_d_gamma)
        return dF

    def free_energy_vqe(self, x0):
        #Run a VQE to minimize x.
        print("Running Gibbs VQE...")
        self.vqe_iteration = 0
        print(f"Iteration   Energy")
        print(f"0   {self.compute_free_energy(x0)}")
        
        res = scipy.optimize.minimize(self.compute_free_energy,
                                      x0,
                                      method = "BFGS",
                                      jac = self.compute_free_energy_gradient,
                                      options = {'gtol': 1e-6, 'disp': True},
                                      callback = self.callback
                                      )        
        return res.fun, list(res.x)

    def gibbs_adapt_vqe(self, max_depth = 20):
        adapt_iteration = 0

        Done = False
        while Done == False:
            adapt_iteration += 1
            print(f"ADAPT Iteration {adapt_iteration}")
            add_grad = self.compute_addition_gradient()
            idx = np.argsort(-abs(add_grad))
            add_grad = add_grad[idx]
            print(f"Adding Operator {idx[0]}")
            self._tops.append(idx[0])
            self._tamps.append(0.0)
            print(f"TOPS: {self._tops}") 
            E, self._tamps = self.free_energy_vqe(self._tamps)                

            if len(self._tops) == max_depth:
                Done = True

        return E, self._tamps

    def callback(self, x):
        self.vqe_iteration += 1
        print(f"{self.vqe_iteration}    {self.compute_free_energy(x)}")
        

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
                   


        



        
