"""
Classes for Gibbs State ADAPT-VQE
====================================
"""

import qforte as qf

from qforte.abc.uccvqeabc import UCCVQE

import numpy as np
import scipy

kb = 3.166811563455546e-06

class Gibbs_ADAPT(UCCVQE):
    def run(self,
            references = None,
            pool_type = "GSD",
            verbose = False,
            T = 273):
        
        self._references = references
        try:
            assert T > 0
        except:
            print("Only T > 0 supported.")
        self.beta = 1/(kb*T)

        self._pool_type = pool_type
        self.verbose = verbose
        self._Upreps = [qf.build_Uprep(ref, 'occupation_list') for ref in references] 
        self.fill_pool() 


    def compute_free_energy(self, x, return_distribution = False):
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
        b_exp_Es = np.array([np.exp(self.beta*(Es[0]-E)) for E in Es])
        tot_b_exp_Es = np.sum(b_exp_Es)
        energy = np.sum(b_exp_Es*Es)/tot_b_exp_Es
        entropy = np.sum((-self.beta*Es + np.ones(Es.shape)*(self.beta*Es[0] - np.log(tot_b_exp_Es)))*b_exp_Es/tot_b_exp_Es)        
        if return_distribution == True:
            return energy + entropy, b_exp_Es/tot_b_exp_Es
        return energy + entropy

    def compute_free_energy_gradient(self, x):
        #Get energy expectation values and gradients.
        Es = []
        grads = []
        for i, reference in enumerate(self._references):
            self._Uprep = self._Upreps[i]
            self.energy_feval(x)
            Es.append(self.energy_feval(x))
            grads.append(self.measure_gradient(x))
        Es = np.array(Es)
        grads = np.array(grads)

        #Re-sort so that E0 is the lowest.
        idx = np.argsort(Es)
        self._Upreps = [self._Upreps[i] for i in idx]
        Es = Es[idx]
        grads = grads[idx]

        #Compute gradients w.r.t. parameters.
        b_exp_Es = np.array([np.exp(self.beta*(Es[0]-E)) for E in Es])
        tot_b_exp_Es = np.sum(b_exp_Es)

        d_b_exp_Es = np.einsum('im,i->im', (np.ones(grads.shape)*grads[0] - grads), self.beta * (np.ones(Es.shape)*Es[0] - Es))
        tot_d_b_exp_Es = np.einsum('im->m',d_b_exp_Es)

        dF = (tot_b_exp_Es * (np.einsum('im,i->m',d_b_exp_Es,Es) + np.einsum('i,im->m',b_exp_Es,grads)) - np.einsum('i,i,m->m', b_exp_Es, Es, tot_d_b_exp_Es))/np.multiply(tot_b_exp_Es,tot_b_exp_Es)
        dF += np.einsum('im,i->m', tot_b_exp_Es*d_b_exp_Es - np.einsum('i,m->im',b_exp_Es,tot_d_b_exp_Es), (self.beta*np.ones(len(Es))*(self.beta*Es[0] - np.log(tot_b_exp_Es))-Es)/np.multiply(tot_b_exp_Es,tot_b_exp_Es))
        dF += (self.beta/tot_b_exp_Es) * np.einsum('i,m->m', b_exp_Es, grads[0,:])
        dF -= (self.beta/tot_b_exp_Es) * np.einsum('i,im->m', b_exp_Es, grads)
        dF -= (1/tot_b_exp_Es**2)*np.einsum('i,m->m', b_exp_Es, tot_d_b_exp_Es)
        return dF
        
    def free_energy_vqe(self, x0):
        #Run a VQE to minimize x.

        res = scipy.optimize.minimize(self.compute_free_energy,
                                      x0,
                                      method = "BFGS",
                                      jac = self.compute_free_energy_gradient,
                                      options = {'gtol': 1e-6, 'disp': True}
                                      )        
        assert res.success == True
        return res.fun, res.x


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
                   


        



        
