from pytest import approx
import qforte as qf
import numpy as np

class TestFCICompApply:
    def test_apply_h4_ham_to_fci_computer(self):

        nel = 4
        sz = 0
        norb = 4

        fci_comp = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
        fci_comp.hartree_fock()

        loaded_h4_ham = np.load('zip_files/h4_sq_ham.npz', allow_pickle=True)['data']
        loaded_h4_hf_sigma = np.load('zip_files/h4_hf_sigma.npz', allow_pickle=True)['data']

        h4_ham = qf.SQOperator()

        # Iterate through the elements and print them
        for element in loaded_h4_ham:
            coeff = element[0]
            cres = element[1]
            anns = element[2]

            h4_ham.add_term(coeff, cres, anns)

        fci_comp.apply_sqop(h4_ham)

        Cdif = qf.Tensor(fci_comp.get_state().shape(), "Cdif")
        Cdif.copy_in(fci_comp.get_state())

        Cfqe = qf.Tensor(fci_comp.get_state().shape(), "Cfqe")
        Cfqe.fill_from_nparray(
            loaded_h4_hf_sigma.ravel(), 
            Cfqe.shape())
        
        Cdif.subtract(Cfqe)

        cdif_norm = Cdif.norm()

        assert cdif_norm == approx(0, abs=1.0e-14)