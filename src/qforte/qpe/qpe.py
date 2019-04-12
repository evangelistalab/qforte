import qforte
import numpy
from qforte.experiment import *
from qforte.utils import *

class QPE(object):

    def __init__(self, n_qubits_measure, n_qubits_phase, operator, N_samples, qpe_algorithm):
        self.n_qubits_measure_ = n_qubits_measure
        self.n_qubits_phase_ = n_qubits_phase
        self.operator_ = operator
        self.N_samples_ = N_samples
        self.qpe_algorithm_ = qpe_algorithm
        self.experiment_ = qforte.Experiment(n_qubits, generator, operator, N_samples)

    def build_control(circ)

        return control_circ

    def generate_qpe_circuit(self):
        
        circ = qforte.QuantumCircuit()

        # 1. Add superposition gates

        # 2. Add control-U sequence

        # 3. Add QFT gates
        
        return circ

    def do_qpe(self):

        self.generator = generate_qpe_circuit()

        result = self.experiment_.experimental_avg

        return result
