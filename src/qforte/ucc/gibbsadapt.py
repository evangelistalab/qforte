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
        
        
        self._pool_type = pool_type
        self._compact_excitations = True
        self.fill_pool() 
        
        self._references = references
        self._weights = [1]*len(references)
        self._is_multi_state = True
        self.T = T
        try:
            assert(self.T > 0)
        except:
            print("T must be positive.")
            exit()

        self.history = []
        self.beta = 1/(kb*T)
        
        
        self.verbose = verbose
        self._Uprep = references
        #self._Upreps = [qf.build_Uprep(ref, 'occupation_list') for ref in references]
        


    def compute_free_energy(self, x, return_U = False):
        #Get energy expectation values. 
        U = self.build_Uvqc(amplitudes = x)
        E, A, ops = qf.ritz_eigh(self._nqb, self._qb_ham, U, [], verbose = False)
        
        q = np.exp(-self.beta * (E - np.ones(E.shape)*E[0]))
        
        Q = np.sum(q)
        p = []
        for state in q:
            if abs(state) > 0:
                p.append(state/Q)
            else:
                p.append(0)
        
        U = E.T@p
        
        S = (1/self.beta)*(-np.sum(p)*np.log(Q))
        for i in range(len(q)):
            if abs(q[i]) > 0:
                S += (1/self.beta*p[i]*np.log(q[i]))
        
        #S = (1/self.beta)*(p.T@np.log(q) - np.sum(p)*np.log(Q))
        self.current_energy = U + S
        if return_U == False:
            return U + S
        else:
            return U + S, U

    def compute_free_energy_gradient(self, x):
        #Get energy expectation values.
        
        U = self.build_Uvqc(amplitudes = x)
        E, A, ops = qf.ritz_eigh(self._nqb, self._qb_ham, U, [], verbose = False)
        E = np.array(E).real
        A = A.real 
        
        dH = self.measure_gradient(params=x, couplings = True)
        dE = np.einsum('ij,jku,ki->iu', A.T, dH, A)
        
        
        q = np.exp(-self.beta * (E - np.ones(E.shape)*E[0]))
        Q = np.sum(q)
        p = q/Q 

        dF = np.einsum('i,iu->u', p, dE)
        '''
        print(f"Analytical gradient: {dF}")
        num_grad = []
        h = 1e-5
        for i in range(len(dF)): 
            forw = copy.copy(x)
            rev = copy.copy(x)
            forw[i] += h
            rev[i] -= h
            forw = self.compute_free_energy(forw)
            rev = self.compute_free_energy(rev)
            num_grad.append((forw-rev)/(2*h))
        num_grad = np.array(num_grad)
        print(f"Numerical gradient: {num_grad}")
        '''
        return dF

    def compute_addition_gradient(self):

        #Get energy expectation values.        
        U = self.build_Uvqc(amplitudes = self._tamps)
        E, A, ops = qf.ritz_eigh(self._nqb, self._qb_ham, U, [], verbose = False)
        A = A.real
        dH = self.measure_gradient3(coupling = True)
        dE = np.einsum('ij,jku,ki->iu', A.T, dH, A)
        
        q = np.exp(-self.beta * (E - np.ones(E.shape)*E[0]))
        Q = np.sum(q)
        p = q/Q 

        dF = np.einsum('i,iu->u', p, dE)        
        
        return dF

    def free_energy_vqe(self, x0):
        #Run a VQE to optimize x.
        print("Running Gibbs VQE...")
        self.vqe_iteration = 0
        print(f"Iteration   Energy")
        self.current_energy = self.compute_free_energy(x0)
        print(f"0 {self.current_energy}")
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
            add_grad = self.compute_addition_gradient()
            idx = np.argsort(-abs(add_grad))
            
            add_grad = add_grad[idx]
            print(f"Adding operator {idx[0]} with gradient {add_grad[0]}")
            self._tops.append(idx[0])
            self._tamps.append(0.0)
            print(f"TOPS: {self._tops}") 
            E, self._tamps = self.free_energy_vqe(self._tamps)
            self.history.append((E, self._tamps))                
            E, U = self.compute_free_energy(self._tamps, return_U = True)
            print(f"ADAPT Energy {adapt_iteration}: {E} {U}", flush = True)
            if len(self._tops) == max_depth:
                Done = True

        return E, self._tamps

    def callback(self, x):
        self.vqe_iteration += 1
        print(f"{self.vqe_iteration} {self.current_energy}", flush = True)
        

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
                   


        



        
