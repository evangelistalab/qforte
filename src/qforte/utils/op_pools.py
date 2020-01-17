"""
oppool.py
====================================
A class for operator management and
consturction.
"""

import qforte
import numpy as np

class SDOpPool(object):
    """
    A class that builds a pools of index lists pertaining to second quatized
    operators to use in VQE and other quantum algorithms. All spin complete,
    particle-hole single and double exccitations are considered.

    Attributes
    ----------
    _pool : list of lists containing a tuple and a float
        The list of singe and double excitation
        operators to consizer. represented in the form,
        [ [(p,q), t_pq], .... , [(p,q,s,r), t_pqrs], ... ]
        where p, q, r, s are idicies of normal ordered creation or anihilation
        operators.

    _ref : list
        The reference state given as a list of 1's and 0's representing spin
        orbital occupations (e.g. the Hartree-Fock state).

    _nqubtis : int
        The number of qubits for the calcuation.

    _nocc : int
        The number of occupied spatial orbtitals to cosider for particle-hole
        formalism (derived from reference if not specified).

    _nvir : int
        The number of unoccupied spatial orbtitals to cosider for particle-hole
        formalism (derived from reference if not specified).


    Methods
    -------
    get_orb_occs()
        Retruns the number of occupied and unoccupied orbitals if nocc and nvir
        are not specified.

    get_singlet_SD_op_pool()
        Retruns a list with indicies pertaining spin-complete, single and
        double excitation operators according to _nocc and _nvir.

    get_canonical_order()
        Takes a subterm and returns the normal ordered verion of it.

    simplify_single_term()
        Takes a list of operator indicies pertainig to a single spin-complete
        operator and normal orders all subterms, then combines like subterms.

    get_simplified_SD_pool()
        Simplifies a list of all spin-complete operators.

    fill_pool()
        Checks the multiplicity and excitation order before calling
        get_singlet_SD_op_pool() to occupy pool_.

    get_pool_lst()
        Returns pool_.

    print_pool()
        Prints all the operators in the pool.

    """

    #TODO: Fix N_samples arg in Experiment class to only be take for finite measurement
    #TODO: Remove order option for SDPool
    def __init__(self, ref, nocc=None, nvir=None, multiplicity = 0, order = 2):
        """
        Parameters
        ----------
        ref : list
            The set of 1s and 0s indicating the initial quantum state.

        nocc : int (optional)
            The number of spatial occupied orbitals.

        nvir : int (optional)
            The number of spatial unoccupied orbitals.

        multiplicity : int
            The targeted multiplicity.

        order : int
            The level of excitation order to use. For exampte order = 2 refers
            to single and double excitatoins.

        """
        #TODO(Nick): Elimenate getting info about nqubits in the 'len(ref)' fashion
        self._ref = ref
        self._nqubtis = len(ref)

        if(nocc==None and nvir==None):
            self._nocc, self._nvir = self.get_orb_occs()
        else:
            self._nocc = nocc
            self._nvir = nvir

    def get_orb_occs(self):
        norb = len(self._ref)
        if (norb%2 == 0):
            norb = int(norb/2)
        else:
            raise NotImplementedError("QForte does not yet support systems with an odd number of spin orbitals.")

        nocc = 0
        for occupancy in self._ref:
            nocc += int(occupancy)

        if (nocc%2 == 0):
            nocc = int(nocc/2)
        else:
            raise NotImplementedError("QForte does not yet support systems with an odd number of occupied spin orbitals.")

        nvir = int(norb - nocc)

        return nocc, nvir

    def get_singlet_SD_op_pool(self):
        """
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        NOTE* ops are not ordered by a - (a)^dag,
        rather they are ordered a + b + c... - (a)^dag - (b)^dag - (c)^dag
        """

        op_pool = []

        for i in range(0,self._nocc):
            ia = 2*i
            ib = 2*i+1

            for a in range(0,self._nvir):
                aa = 2*self._nocc + 2*a
                ab = 2*self._nocc + 2*a+1

                temp1 = []
                temp1.append([ (aa, ia), 1/np.sqrt(2) ])
                temp1.append([ (ab, ib), 1/np.sqrt(2) ])

                temp1.append([ (ia, aa), -1/np.sqrt(2) ])
                temp1.append([ (ib, ab), -1/np.sqrt(2) ])

                temp1_coeff = 0.0
                for term in temp1:
                    coeff_t = term[1]
                    temp1_coeff += coeff_t * coeff_t

                for term in temp1:
                    term[1] = term[1]/np.sqrt(temp1_coeff)

                op_pool.append(temp1)


        for i in range(0,self._nocc):
            ia = 2*i
            ib = 2*i+1

            for j in range(i,self._nocc):
                ja = 2*j
                jb = 2*j+1

                for a in range(0,self._nvir):
                    aa = 2*self._nocc + 2*a
                    ab = 2*self._nocc + 2*a+1

                    for b in range(a,self._nvir):
                        ba = 2*self._nocc + 2*b
                        bb = 2*self._nocc + 2*b+1

                        temp2a = []
                        if((aa != ba) and (ia != ja)):
                            temp2a.append([ (aa,ba,ia,ja), 2/np.sqrt(12) ])

                        if((ab != bb ) and (ib != jb)):
                            temp2a.append([ (ab,bb,ib,jb), 2/np.sqrt(12) ])

                        if((aa != bb) and (ia != jb)):
                            temp2a.append([ (aa,bb,ia,jb), 1/np.sqrt(12) ])

                        if((ab != ba) and (ib != ja)):
                            temp2a.append([ (ab,ba,ib,ja), 1/np.sqrt(12) ])

                        if((aa != bb) and (ib != ja)):
                            temp2a.append([ (aa,bb,ib,ja), 1/np.sqrt(12) ])

                        if((ab != ba) and (ia != jb)):
                            temp2a.append([ (ab,ba,ia,jb), 1/np.sqrt(12) ])

                        # Hermetian conjugate
                        if((ja != ia) and (ba != aa)):
                            temp2a.append([ (ja,ia,ba,aa), -2/np.sqrt(12) ])

                        if((jb != ib ) and (bb != ab)):
                            temp2a.append([ (jb,ib,bb,ab), -2/np.sqrt(12) ])

                        if((jb != ia) and (bb != aa)):
                            temp2a.append([ (jb,ia,bb,aa), -1/np.sqrt(12) ])

                        if((ja != ib) and (ba != ab)):
                            temp2a.append([ (ja,ib,ba,ab), -1/np.sqrt(12) ])

                        if((ja != ib) and (bb != aa)):
                            temp2a.append([ (ja,ib,bb,aa), -1/np.sqrt(12) ])

                        if((jb != ia) and (ba != ab)):
                            temp2a.append([ (jb,ia,ba,ab), -1/np.sqrt(12) ])


                        temp2b = []
                        if((aa != bb) and (ia != jb)):
                            temp2b.append([ (aa,bb,ia,jb),  0.5 ])

                        if((ab != ba) and (ib != ja)):
                            temp2b.append([ (ab,ba,ib,ja),  0.5 ])

                        if((aa != bb) and (ib != ja)):
                            temp2b.append([ (aa,bb,ib,ja), -0.5 ])

                        if((ab != ba) and (ia != jb)):
                            temp2b.append([ (ab,ba,ia,jb), -0.5 ])

                        # Hermetian conjugate
                        if((jb != ia) and (bb != aa)):
                            temp2b.append([ (jb,ia,bb,aa),  -0.5 ])

                        if((ja != ib) and (ba != ab)):
                            temp2b.append([ (ja,ib,ba,ab),  -0.5 ])

                        if((ja != ib) and (bb != aa)):
                            temp2b.append([ (ja,ib,bb,aa),  0.5 ])

                        if((jb != ia) and (ba != ab)):
                            temp2b.append([ (jb,ia,ba,ab),  0.5 ])

                        #Normalize
                        temp2a_coeff = 0.0
                        temp2b_coeff = 0.0
                        for term in temp2a:
                            coeff_t = term[1]
                            temp2a_coeff += coeff_t * coeff_t
                        for term in temp2b:
                            coeff_t = term[1]
                            temp2b_coeff += coeff_t * coeff_t

                        for term in temp2a:
                            term[1] = term[1]/np.sqrt(temp2a_coeff)
                        for term in temp2b:
                            term[1] = term[1]/np.sqrt(temp2b_coeff)

                        op_pool.append(temp2a)
                        op_pool.append(temp2b)

        return op_pool

    def get_canonical_order(self, sub_term):
        """
        takes in [(4, 7, 0, 1), +0.3535533905932738]
        returns  [(7, 4, 1, 0), +0.3535533905932738]
        i.e.     [(big->small, big->small), incorparate sign flips  ]

        Parameters
        ----------
        sub_term : list of tuple and float
            A list specifying the subterm to be normal ordered.
        """

        nbody = int ( len(sub_term[0]) / 2 )

        if nbody < 1:
            raise ValueError("Can't have zero body operators!")
        if nbody == 1:
            return sub_term

        if nbody == 2:
            new_coeff = sub_term[1]

            lop = list(sub_term[0][:nbody])
            rop = list(sub_term[0][nbody:])

            lop_s = sorted(lop, reverse = True)
            rop_s = sorted(rop, reverse = True)

            if lop != lop_s:
                new_coeff *= -1.0

            if rop != rop_s:
                new_coeff *= -1.0

            new_subterm = [ tuple( lop_s + rop_s ), new_coeff ]
            return new_subterm

        else:
            raise NotImplementedError("Canonical ordering for higher than 2-body terms not avalable")

    def simplify_single_term(self, ops):
        """
        Parameters
        ----------
        ops : list list lists of tuple and float
            A list of subterms in one spin-complete operator.
        """
        can_ordered_ops = []
        for sub_term in ops:
            can_ordered_ops.append(self.get_canonical_order(sub_term))

        term_coeff_dict = {}
        combined_sq_organizer = []
        threshold = 1.0e-10

        for op, coeff in can_ordered_ops:
            temp_tup = tuple(op)
            if( temp_tup not in term_coeff_dict.keys() ):
                term_coeff_dict[temp_tup] = coeff
            else:
                term_coeff_dict[temp_tup] += coeff

        for op in term_coeff_dict:
            if(np.abs(term_coeff_dict[op]) > threshold):
                combined_sq_organizer.append([tuple(op),
                                            term_coeff_dict[op]])

        return combined_sq_organizer

    def get_simplified_SD_pool(self):
        unsimplified_op_lst = self.get_singlet_SD_op_pool()
        simplified_op_list = []

        for k, term in enumerate(unsimplified_op_lst):
            single_term = self.simplify_single_term(term)
            if (len(single_term) > 0):
                simplified_op_list.append(single_term)

        #Normalize
        temp_coeff_lst = []
        for op in simplified_op_list:
            temp_coeff = 0.0
            for sub_term in op:
                coeff_t = sub_term[1]
                temp_coeff += coeff_t * coeff_t
            temp_coeff_lst.append(temp_coeff)

        for k, op in enumerate(simplified_op_list):
            for sub_term in op:
                sub_term[1] = sub_term[1]/np.sqrt(temp_coeff_lst[k])

        return simplified_op_list

    def fill_pool(self, multiplicity = 0, order = 2):
        """
        Parameters
        ----------
        multiplicity : int
            The targeted multiplicity.

        order : int
            The level of excitation order to use. For exampte order = 2 refers
            to single and double excitatoins.
        """

        if (multiplicity != 0) or (order != 2):
            raise NotImplementedError("Qforte currently supports only singlet singles and doubles for pool construction.")

        self._pool = self.get_simplified_SD_pool()

    def get_pool_lst(self):
        return self._pool

    def print_pool(self):
        print('\n\n-------------------------------------')
        print('  Singles and Doubles operator pool')
        print('-------------------------------------')
        for k, op in enumerate(self._pool):
            print('\n----> ', k, ' <----')
            for term in op:
                print(term)
        print('\n')
