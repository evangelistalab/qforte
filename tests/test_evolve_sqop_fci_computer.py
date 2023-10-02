from pytest import approx
import qforte as qf
import numpy as np

class TestFCICompApply:
    def test_evolve_sqop_from_hf_fci_computer(self):

        nel = 4
        sz = 0
        norb = 4

        fci_comp = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
        fci_comp.hartree_fock()

        loaded_c_fqe = np.load('zip_files/4e_4o_small_sqop_evo_from_hf.npz', allow_pickle=True)['data']
        Cfqe = qf.Tensor(fci_comp.get_state().shape(), "Cfqe")
        Cfqe.fill_from_nparray(
            loaded_c_fqe.ravel(), 
            Cfqe.shape())

        sq_terms = [
            (+0.704645 * 1.0j, [7, 6], [3, 2]), # 2body ab 
            (+0.4 * 1.0j, [6], [0]), # 1bdy-a
            (+0.4 * 1.0j, [7], [3]), # 1bdy-a
            (+0.704645 * 1.0, [6, 3], [3, 2]), # 2body-nbr ab 
            (+0.704645 * 1.0, [6, 5], [5, 2]), # 2body-nbr ab (coeff must be REAL)
            (+0.704645 * 1.0, [2], [2]), # 1body-nbr ab (coeff must be REAL)
            ]

        time = 1.0

        for sq_term in sq_terms:
            sqop = qf.SQOperator()
            sqop.add_term(sq_term[0], sq_term[1], sq_term[2])
            sqop.add_term(np.conj(sq_term[0]), sq_term[2][::-1], sq_term[1][::-1])
            fci_comp.apply_sqop_evolution(time, sqop)


        Cdif = qf.Tensor(fci_comp.get_state().shape(), "Cdif")
        Cdif.copy_in(fci_comp.get_state())
        norm = Cdif.norm()

        Cdif.subtract(Cfqe)
        cdif_norm = Cdif.norm()

        assert norm == approx(1.0, abs=1.0e-14)
        assert cdif_norm == approx(0.0, abs=1.0e-14)

        # print(Cdif.str(print_complex=True))
        # print(Cfqe.str(print_complex=True))

    def test_evolve_sqop_from_random_fci_computer(self):

        nel = 4
        sz = 0
        norb = 4

        loaded_inital_state = np.load(
            'zip_files/4e_4o_random_starting_state.npz', 
            allow_pickle=True)['data']

        fci_comp = qf.FCIComputer(nel=nel, sz=sz, norb=norb)

        Co = qf.Tensor(fci_comp.get_state().shape(), "Cdif")
        Co.fill_from_nparray(
            loaded_inital_state.ravel(), 
            Co.shape())
        
        co_norm = Co.norm()
        
        fci_comp.set_state(Co)

        print(fci_comp)

        loaded_c_fqe = np.load('zip_files/4e_4o_small_sqop_evo_from_random.npz', allow_pickle=True)['data']
        Cfqe = qf.Tensor(fci_comp.get_state().shape(), "Cfqe")
        Cfqe.fill_from_nparray(
            loaded_c_fqe.ravel(), 
            Cfqe.shape())

        sq_terms = [
            (+0.704645 * 1.0j, [7, 6], [3, 2]), # 2body ab 
            (+0.4 * 1.0j, [6], [0]), # 1bdy-a
            (+0.4 * 1.0j, [7], [3]), # 1bdy-a
            (+0.704645 * 1.0, [6, 3], [3, 2]), # 2body-nbr ab 
            (+0.704645 * 1.0, [6, 5], [5, 2]), # 2body-nbr ab (coeff must be REAL)
            (+0.704645 * 1.0, [2], [2]), # 1body-nbr ab (coeff must be REAL)
            ]

        time = 1.0

        for sq_term in sq_terms:
            sqop = qf.SQOperator()
            sqop.add_term(sq_term[0], sq_term[1], sq_term[2])
            sqop.add_term(np.conj(sq_term[0]), sq_term[2][::-1], sq_term[1][::-1])
            fci_comp.apply_sqop_evolution(time, sqop)


        Cdif = qf.Tensor(fci_comp.get_state().shape(), "Cdif")
        Cdif.copy_in(fci_comp.get_state())
        norm = Cdif.norm()

        Cdif.subtract(Cfqe)
        cdif_norm = Cdif.norm()

        assert norm == approx(1.0, abs=1.0e-14)
        assert cdif_norm == approx(0.0, abs=1.0e-14)

        print(f"||Co||:   {co_norm}")
        print(f"||Cf||:   {norm}")
        print(f"||Cdif||: {cdif_norm}")

        print(Cdif.str(print_complex=True))
        print(Cfqe.str(print_complex=True))