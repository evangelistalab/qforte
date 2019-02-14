#include <stdexcept>

#include "fmt/format.h"

#include "quantum_gate.h"

QuantumGate make_gate(std::string type, size_t target, size_t control, double parameter) {
    using namespace std::complex_literals;
    if (target == control) {
        if (type == "X") {
            std::complex<double> gate[4][4]{
                {0.0, 1.0},
                {1.0, 0.0},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "Y") {
            std::complex<double> gate[4][4]{
                {0.0, -1.0i},
                {+1.0i, 0.0},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "Z") {
            std::complex<double> gate[4][4]{
                {+1.0, 0.0},
                {0.0, -1.0},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "H") {
            std::complex<double> c = 1.0 / std::sqrt(2.0);
            std::complex<double> gate[4][4]{
                {+c, +c},
                {+c, -c},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "Rzy") {
            std::complex<double> c = 1.0 / std::sqrt(2.0);
	    std::complex<double> ci = 1.0i / std::sqrt(2.0);
            std::complex<double> gate[4][4]{
                {+c, +ci},
                {+c, -ci},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "R") {
	    std::complex<double> tmp = 1.0i * parameter;
            std::complex<double> c = std::exp(tmp);
            std::complex<double> gate[4][4]{
                {1.0, 0.0},
                {0.0, c},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "S") {
            std::complex<double> gate[4][4]{
                {1.0, 0.0},
                {0.0, 1.0i},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "T") {
            std::complex<double> c = 1.0 / std::sqrt(2.0);
	    std::complex<double> tmp = 1.0 + 1.0i;
            std::complex<double> gate[4][4]{
                {1.0, 0.0},
                {0.0, c * tmp},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "I") {
            std::complex<double> gate[4][4]{
                {1.0, 0.0},
                {0.0, 1.0},
            };
            return QuantumGate(type, target, control, gate);
        }
    } else {
        if ((type == "cX") or (type == "CNOT")) {
            std::complex<double> gate[4][4]{
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 1.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 1.0},
                {0.0, 0.0, 1.0, 0.0},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "cY") {
            std::complex<double> gate[4][4]{
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 1.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, -1.0i},
                {0.0, 0.0, +1.0i, 0.0},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "cZ") {
            std::complex<double> gate[4][4]{
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 1.0, 0.0, 0.0},
                {0.0, 0.0, 1.0, 0.0},
                {0.0, 0.0, 0.0, -1.0},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "SWAP") {
            std::complex<double> gate[4][4]{
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 1.0, 0.0},
                {0.0, 1.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 1.0},
            };
            return QuantumGate(type, target, control, gate);
        }
    }
    // If you reach this section then the gate type is not implemented or it is invalid.
    // So we throw an exception that propagates to Python and return the identity
    std::string msg =
        fmt::format("make_quantum_gate()\ntype = {} is not a valid quantum gate type", type);
    throw std::invalid_argument(msg);
    std::complex<double> gate[4][4]{
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 1.0},
    };
    return QuantumGate(type, target, control, gate);
}
